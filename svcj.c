#include "svcj.h"
#include <float.h>

// --- Utilities ---

void clean_returns(double* returns, int n) {
    for(int i=0; i<n; i++) {
        if(fabs(returns[i]) < 1e-12) {
            returns[i] = (i % 2 == 0) ? 1e-12 : -1e-12;
        }
    }
}

// Compute ln(C_t / C_{t-1}) from OHLCV matrix
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) {
    // ohlcv is flat array: row i, col j -> ohlcv[i*N_COLS + j]
    // returns size is n_rows - 1
    for(int i=1; i<n_rows; i++) {
        double prev_c = ohlcv[(i-1)*N_COLS + IDX_CLOSE];
        double curr_c = ohlcv[i*N_COLS + IDX_CLOSE];
        if(prev_c < 1e-9) prev_c = 1e-9;
        out_returns[i-1] = log(curr_c / prev_c);
    }
}

void check_constraints(SVCJParams* p) {
    if(p->kappa < 0.2) p->kappa = 0.2;     if(p->kappa > 30.0) p->kappa = 30.0;
    if(p->theta < 0.0001) p->theta = 0.0001; if(p->theta > 1.0) p->theta = 1.0; 
    if(p->sigma_v < 0.05) p->sigma_v = 0.05; if(p->sigma_v > 5.0) p->sigma_v = 5.0;
    if(p->rho > 0.95) p->rho = 0.95;       if(p->rho < -0.95) p->rho = -0.95;
    if(p->lambda_j < 0.05) p->lambda_j = 0.05; if(p->lambda_j > 252.0) p->lambda_j = 252.0; 
    if(p->sigma_j < 0.005) p->sigma_j = 0.005; if(p->sigma_j > 0.3) p->sigma_j = 0.3; 
    
    // Feller Soft
    double feller = 2.0 * p->kappa * p->theta;
    if (p->sigma_v * p->sigma_v > feller * 4.0) p->sigma_v = sqrt(feller * 4.0);
}

// --- Robust Initialization using Garman-Klass (High-Low) ---
// This uses OHLCV to find a better 'Theta' than just Close-Close variance
void estimate_initial_params_ohlcv(double* ohlcv, int n, SVCJParams* p) {
    double sum_gk = 0.0;
    
    // Garman-Klass Volatility Estimator
    // 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
    for(int i=0; i<n; i++) {
        double O = ohlcv[i*N_COLS + IDX_OPEN];
        double H = ohlcv[i*N_COLS + IDX_HIGH];
        double L = ohlcv[i*N_COLS + IDX_LOW];
        double C = ohlcv[i*N_COLS + IDX_CLOSE];
        
        if(L < 1e-9) L = 1e-9; if(O < 1e-9) O = 1e-9;
        
        double hl = log(H/L);
        double co = log(C/O);
        double val = 0.5 * hl*hl - (2.0*log(2.0)-1.0) * co*co;
        if(val > 0) sum_gk += val;
    }
    
    double mean_gk = sum_gk / n;
    double rv_annual = mean_gk * 252.0;

    p->mu = 0.0;
    p->theta = rv_annual; 
    if(p->theta < 0.0025) p->theta = 0.0025;
    
    p->kappa = 4.0;
    p->sigma_v = sqrt(p->theta); 
    p->rho = -0.6;
    p->lambda_j = 0.5; 
    p->mu_j = 0.0;
    p->sigma_j = sqrt(rv_annual/252.0) * 4.0; 
    
    check_constraints(p);
}

// --- UKF Core ---
double ukf_log_likelihood(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob, double realized_theta_proxy) {
    double ll = 0.0;
    double v = p->theta; 
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        // Predict
        double v_pred = v + p->kappa * (p->theta - v) * DT;
        if(v_pred < 1e-6) v_pred = 1e-6;

        double drift = (p->mu - 0.5 * v_pred);
        double y = returns[t] - drift * DT;

        // Phenotypic Mixing: Robust Variance Floor
        // Mix instantaneous prediction with Long-Run mean to prevent collapse
        double robust_var_d = fmax(v_pred, 0.25 * p->theta) * DT; 
        
        // Likelihoods
        double pdf_d = (1.0 / (sqrt(robust_var_d) * SQRT_2PI)) * exp(-0.5 * y*y / robust_var_d);
        
        double total_var_j = robust_var_d + var_j; 
        double y_j = y - p->mu_j; 
        double pdf_j = (1.0 / (sqrt(total_var_j) * SQRT_2PI)) * exp(-0.5 * y_j*y_j / total_var_j);
        
        // Posterior
        double prob_prior = p->lambda_j * DT;
        if(prob_prior > 0.9) prob_prior = 0.9;
        
        double num = pdf_j * prob_prior;
        double den = num + pdf_d * (1.0 - prob_prior);
        if(den < 1e-15) den = 1e-15;
        double prob_posterior = num / den;
        
        // Update
        double S = v_pred * DT + prob_posterior * var_j;
        double K = (p->rho * p->sigma_v * DT) / S;
        v = v_pred + K * y;
        
        if(v < 1e-6) v = 1e-6;
        if(v > 4.0) v = 4.0;

        if(out_spot_vol) out_spot_vol[t] = sqrt(v_pred); 
        if(out_jump_prob) out_jump_prob[t] = prob_posterior;

        ll += log(den); 
    }
    
    // MAP Penalties
    // Penalty 1: Theta shouldn't deviate too far from Garman-Klass proxy
    double theta_penalty = -50.0 * pow(log(p->theta) - log(realized_theta_proxy), 2);
    
    // Penalty 2: Jump Size validity
    double target_jump = sqrt(realized_theta_proxy/252.0) * 3.0;
    double jump_penalty = -20.0 * pow(log(p->sigma_j) - log(target_jump), 2);
    
    double rho_penalty = -2.0 * pow(p->rho + 0.5, 2);

    if(isnan(ll) || isinf(ll)) return -1e15;
    return ll + theta_penalty + jump_penalty + rho_penalty;
}

// --- Optimization ---
void optimize_svcj(double* ohlcv, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    // 1. Estimate Start Params using OHLCV (Garman-Klass)
    estimate_initial_params_ohlcv(ohlcv, n, p);
    double theta_proxy = p->theta; // Keep this as anchor
    
    // 2. Compute Returns Vector for UKF (Close-to-Close)
    // Note: n rows of data = n-1 returns
    int n_ret = n - 1;
    double* returns = (double*)malloc(n_ret * sizeof(double));
    if(!returns) return;
    
    compute_log_returns(ohlcv, n, returns);
    clean_returns(returns, n_ret);

    // 3. Nelder-Mead
    int n_dim = 5;
    double simplex[6][5];
    double scores[6];
    
    for(int i=0; i<=n_dim; i++) {
        SVCJParams temp = *p;
        if(i==1) temp.kappa *= 1.25;
        if(i==2) temp.theta *= 1.25;
        if(i==3) temp.sigma_v *= 1.25;
        if(i==4) temp.rho = (temp.rho > 0) ? temp.rho - 0.25 : temp.rho + 0.25;
        if(i==5) temp.lambda_j *= 1.5;
        check_constraints(&temp);
        
        simplex[i][0] = temp.kappa; simplex[i][1] = temp.theta; simplex[i][2] = temp.sigma_v;
        simplex[i][3] = temp.rho;   simplex[i][4] = temp.lambda_j;
        scores[i] = ukf_log_likelihood(returns, n_ret, &temp, NULL, NULL, theta_proxy);
    }
    
    for(int iter=0; iter<NM_ITER; iter++) {
        int vs[6]; for(int k=0; k<6; k++) vs[k]=k;
        for(int i=0; i<6; i++) {
            for(int j=i+1; j<6; j++) {
                if(scores[vs[j]] > scores[vs[i]]) { int t=vs[i]; vs[i]=vs[j]; vs[j]=t; }
            }
        }
        
        double c[5] = {0};
        for(int i=0; i<5; i++) { for(int k=0; k<5; k++) c[k] += simplex[vs[i]][k]; }
        for(int k=0; k<5; k++) c[k] /= 5.0;
        
        double ref[5]; SVCJParams rp = *p;
        for(int k=0; k<5; k++) ref[k] = c[k] + 1.0 * (c[k] - simplex[vs[5]][k]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        check_constraints(&rp);
        double r_score = ukf_log_likelihood(returns, n_ret, &rp, NULL, NULL, theta_proxy);
        
        if(r_score > scores[vs[0]]) {
            double exp[5]; SVCJParams ep = *p;
            for(int k=0; k<5; k++) exp[k] = c[k] + 2.0 * (c[k] - simplex[vs[5]][k]);
            ep.kappa=exp[0]; ep.theta=exp[1]; ep.sigma_v=exp[2]; ep.rho=exp[3]; ep.lambda_j=exp[4];
            check_constraints(&ep);
            double e_score = ukf_log_likelihood(returns, n_ret, &ep, NULL, NULL, theta_proxy);
            if(e_score > r_score) { for(int k=0; k<5; k++) simplex[vs[5]][k] = exp[k]; scores[vs[5]] = e_score; } 
            else { for(int k=0; k<5; k++) simplex[vs[5]][k] = ref[k]; scores[vs[5]] = r_score; }
        } else if(r_score > scores[vs[4]]) {
             for(int k=0; k<5; k++) simplex[vs[5]][k] = ref[k]; scores[vs[5]] = r_score;
        } else {
            double con[5]; SVCJParams cp = *p;
            for(int k=0; k<5; k++) con[k] = c[k] + 0.5 * (simplex[vs[5]][k] - c[k]);
            cp.kappa=con[0]; cp.theta=con[1]; cp.sigma_v=con[2]; cp.rho=con[3]; cp.lambda_j=con[4];
            check_constraints(&cp);
            double c_score = ukf_log_likelihood(returns, n_ret, &cp, NULL, NULL, theta_proxy);
            if(c_score > scores[vs[5]]) { for(int k=0; k<5; k++) simplex[vs[5]][k] = con[k]; scores[vs[5]] = c_score; }
            else {
                for(int i=1; i<6; i++) {
                    int idx = vs[i]; SVCJParams sp = *p;
                    for(int k=0; k<5; k++) simplex[idx][k] = simplex[vs[0]][k] + 0.5 * (simplex[idx][k] - simplex[vs[0]][k]);
                    sp.kappa=simplex[idx][0]; sp.theta=simplex[idx][1]; sp.sigma_v=simplex[idx][2]; sp.rho=simplex[idx][3]; sp.lambda_j=simplex[idx][4];
                    check_constraints(&sp); scores[idx] = ukf_log_likelihood(returns, n_ret, &sp, NULL, NULL, theta_proxy);
                }
            }
        }
    }
    
    int best = 0; for(int i=1; i<6; i++) if(scores[i] > scores[best]) best = i;
    p->kappa = simplex[best][0]; p->theta = simplex[best][1]; p->sigma_v = simplex[best][2];
    p->rho = simplex[best][3]; p->lambda_j = simplex[best][4];
    
    // Output Population
    ukf_log_likelihood(returns, n_ret, p, out_spot_vol, out_jump_prob, theta_proxy);
    
    free(returns);
}

// --- Pricing (Unchanged) ---
double normal_cdf(double x) { return 0.5 * erfc(-x * M_SQRT1_2); }
double bs_calc(double S, double K, double T, double r, double v, int type) {
    if(T < 1e-4) return (type==1)?fmax(S-K,0):fmax(K-S,0);
    double d1 = (log(S/K)+(r+0.5*v*v)*T)/(v*sqrt(T));
    double d2 = d1 - v*sqrt(T);
    if(type==1) return S*normal_cdf(d1)-K*exp(-r*T)*normal_cdf(d2);
    else return K*exp(-r*T)*normal_cdf(-d2)-S*normal_cdf(-d1);
}

void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* p, double spot_vol, double* out_prices) {
    double lambda = p->lambda_j; 
    double m = p->mu_j; 
    double v_j = p->sigma_j;
    double lamp = lambda*(1.0+m);
    
    for(int i=0; i<n_opts; i++) {
        double val = 0.0;
        for(int k=0; k<12; k++) {
             double fact=1.0; for(int j=1;j<=k;j++) fact*=j;
             double prob = exp(-lamp*expiries[i]) * pow(lamp*expiries[i], k) / fact;
             if(prob < 1e-8 && k>2) break;
             double rk = p->mu - lambda*m + (k*log(1.0+m))/expiries[i];
             double vk = sqrt(spot_vol*spot_vol + (k*v_j*v_j)/expiries[i]);
             val += prob * bs_calc(s0, strikes[i], expiries[i], rk, vk, types[i]);
        }
        out_prices[i] = val;
    }
}