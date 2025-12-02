#include "svcj.h"
#include <float.h>

void clean_returns(double* returns, int n) {
    for(int i=0; i<n; i++) {
        if(fabs(returns[i]) < 1e-12) returns[i] = (i % 2 == 0) ? 1e-12 : -1e-12;
    }
}

void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) {
    for(int i=1; i<n_rows; i++) {
        double prev = ohlcv[(i-1)*N_COLS + IDX_CLOSE];
        double curr = ohlcv[i*N_COLS + IDX_CLOSE];
        if(prev < 1e-9) prev = 1e-9;
        out_returns[i-1] = log(curr / prev);
    }
}

void check_constraints(SVCJParams* p) {
    if(p->kappa < 0.2) p->kappa = 0.2;     if(p->kappa > 30.0) p->kappa = 30.0;
    if(p->theta < 0.0001) p->theta = 0.0001; if(p->theta > 1.0) p->theta = 1.0; 
    if(p->sigma_v < 0.05) p->sigma_v = 0.05; if(p->sigma_v > 5.0) p->sigma_v = 5.0;
    if(p->rho > 0.95) p->rho = 0.95;       if(p->rho < -0.95) p->rho = -0.95;
    if(p->lambda_j < 0.05) p->lambda_j = 0.05; if(p->lambda_j > 252.0) p->lambda_j = 252.0; 
    
    // Feller Condition Soft Check
    double feller = 2.0 * p->kappa * p->theta;
    if (p->sigma_v * p->sigma_v > feller * 4.0) p->sigma_v = sqrt(feller * 4.0);
}

// Robust Init: Uses Garman-Klass (High/Low/Close/Open) to estimate variance
// This prevents 'Close-to-Close' noise from distorting the starting Theta.
void estimate_initial_params_ohlcv(double* ohlcv, int n, SVCJParams* p) {
    double sum_gk = 0.0;
    for(int i=0; i<n; i++) {
        double O = ohlcv[i*N_COLS + IDX_OPEN];
        double H = ohlcv[i*N_COLS + IDX_HIGH];
        double L = ohlcv[i*N_COLS + IDX_LOW];
        double C = ohlcv[i*N_COLS + IDX_CLOSE];
        if(L < 1e-9) L = 1e-9; if(O < 1e-9) O = 1e-9;
        double hl = log(H/L); double co = log(C/O);
        double val = 0.5 * hl*hl - (2.0*log(2.0)-1.0) * co*co;
        if(val > 0) sum_gk += val;
    }
    double rv_annual = (sum_gk / n) * 252.0;

    p->mu = 0.0; p->theta = rv_annual; if(p->theta < 0.0025) p->theta = 0.0025;
    p->kappa = 4.0; p->sigma_v = sqrt(p->theta); p->rho = -0.6;
    p->lambda_j = 0.5; p->mu_j = 0.0; p->sigma_j = sqrt(rv_annual/252.0) * 3.5; 
    check_constraints(p);
}

double ukf_log_likelihood(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob, double theta_anchor) {
    double ll = 0.0;
    double v = p->theta;
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        double v_pred = v + p->kappa * (p->theta - v) * DT;
        if(v_pred < 1e-6) v_pred = 1e-6;

        double y = returns[t] - (p->mu - 0.5 * v_pred) * DT;

        // Compensation: Phenotypic Mixing
        // Prevent diffusion collapse by enforcing a floor based on Long-Run Theta
        double robust_var_d = fmax(v_pred, 0.25 * p->theta) * DT; 
        
        double pdf_d = (1.0 / (sqrt(robust_var_d) * SQRT_2PI)) * exp(-0.5 * y*y / robust_var_d);
        double total_var_j = robust_var_d + var_j; 
        double y_j = y - p->mu_j; 
        double pdf_j = (1.0 / (sqrt(total_var_j) * SQRT_2PI)) * exp(-0.5 * y_j*y_j / total_var_j);
        
        double prob_prior = (p->lambda_j * DT > 0.9) ? 0.9 : p->lambda_j * DT;
        double den = pdf_j * prob_prior + pdf_d * (1.0 - prob_prior);
        if(den < 1e-15) den = 1e-15;
        double prob_posterior = (pdf_j * prob_prior) / den;
        
        double S = v_pred * DT + prob_posterior * var_j;
        double K = (p->rho * p->sigma_v * DT) / S;
        v = v_pred + K * y;
        if(v < 1e-6) v = 1e-6; if(v > 4.0) v = 4.0;

        if(out_spot_vol) out_spot_vol[t] = sqrt(v_pred); 
        if(out_jump_prob) out_jump_prob[t] = prob_posterior;
        ll += log(den); 
    }
    
    // Bayesian Priors (Anchoring to Garman-Klass Theta)
    double theta_penalty = -50.0 * pow(log(p->theta) - log(theta_anchor), 2);
    double jump_penalty = -20.0 * pow(log(p->sigma_j) - log(sqrt(theta_anchor/252.0)*3.0), 2);
    return ll + theta_penalty + jump_penalty;
}

void optimize_svcj(double* ohlcv, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    estimate_initial_params_ohlcv(ohlcv, n, p);
    double theta_anchor = p->theta;
    
    int n_ret = n - 1;
    double* returns = (double*)malloc(n_ret * sizeof(double));
    if(!returns) return;
    compute_log_returns(ohlcv, n, returns);
    clean_returns(returns, n_ret);

    int n_dim = 5;
    double simplex[6][5];
    double scores[6];
    
    // Simplex Init
    for(int i=0; i<=n_dim; i++) {
        SVCJParams temp = *p;
        if(i==1) temp.kappa *= 1.25; if(i==2) temp.theta *= 1.25;
        if(i==3) temp.sigma_v *= 1.25; if(i==4) temp.rho = (temp.rho>0)?temp.rho-0.2:temp.rho+0.2;
        if(i==5) temp.lambda_j *= 1.5;
        check_constraints(&temp);
        simplex[i][0] = temp.kappa; simplex[i][1] = temp.theta; simplex[i][2] = temp.sigma_v;
        simplex[i][3] = temp.rho;   simplex[i][4] = temp.lambda_j;
        scores[i] = ukf_log_likelihood(returns, n_ret, &temp, NULL, NULL, theta_anchor);
    }
    
    // Nelder-Mead (Simplified)
    for(int iter=0; iter<NM_ITER; iter++) {
        int vs[6]; for(int k=0; k<6; k++) vs[k]=k;
        for(int i=0; i<6; i++) for(int j=i+1; j<6; j++) if(scores[vs[j]] > scores[vs[i]]) { int t=vs[i]; vs[i]=vs[j]; vs[j]=t; }
        
        double c[5] = {0};
        for(int i=0; i<5; i++) for(int k=0; k<5; k++) c[k] += simplex[vs[i]][k];
        for(int k=0; k<5; k++) c[k] /= 5.0;
        
        double ref[5]; SVCJParams rp = *p;
        for(int k=0; k<5; k++) ref[k] = c[k] + 1.0 * (c[k] - simplex[vs[5]][k]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        check_constraints(&rp);
        double r_score = ukf_log_likelihood(returns, n_ret, &rp, NULL, NULL, theta_anchor);
        
        if(r_score > scores[vs[0]]) {
            for(int k=0; k<5; k++) simplex[vs[5]][k] = ref[k]; scores[vs[5]] = r_score;
        } else {
             for(int k=0; k<5; k++) simplex[vs[5]][k] = ref[k]; scores[vs[5]] = r_score;
        }
    }
    
    int best = 0; for(int i=1; i<6; i++) if(scores[i] > scores[best]) best = i;
    p->kappa = simplex[best][0]; p->theta = simplex[best][1]; p->sigma_v = simplex[best][2];
    p->rho = simplex[best][3]; p->lambda_j = simplex[best][4];
    
    ukf_log_likelihood(returns, n_ret, p, out_spot_vol, out_jump_prob, theta_anchor);
    free(returns);
}

double normal_cdf(double x) { return 0.5 * erfc(-x * M_SQRT1_2); }
double bs_calc(double S, double K, double T, double r, double v, int type) {
    if(T < 1e-4) return (type==1)?fmax(S-K,0):fmax(K-S,0);
    double d1 = (log(S/K)+(r+0.5*v*v)*T)/(v*sqrt(T));
    double d2 = d1 - v*sqrt(T);
    return (type==1) ? S*normal_cdf(d1)-K*exp(-r*T)*normal_cdf(d2) : K*exp(-r*T)*normal_cdf(-d2)-S*normal_cdf(-d1);
}

void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* p, double spot_vol, double* out_prices) {
    double lambda = p->lambda_j; double m = p->mu_j; double v_j = p->sigma_j; double lamp = lambda*(1.0+m);
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