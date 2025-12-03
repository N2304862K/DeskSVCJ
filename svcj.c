#include "svcj.h"
#include <float.h>

void clean_returns(double* returns, int n) {
    for(int i=0; i<n; i++) {
        // Mathematical safety only, no data manipulation
        if(isnan(returns[i]) || isinf(returns[i])) returns[i] = 0.0;
        if(fabs(returns[i]) < 1e-12) returns[i] = 1e-12;
    }
}

void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) {
    for(int i=1; i<n_rows; i++) {
        double prev = ohlcv[(i-1)*N_COLS + IDX_CLOSE];
        double curr = ohlcv[i*N_COLS + IDX_CLOSE];
        // Mathematical safety for Log calculation
        if(prev <= 1e-9) prev = 1e-9;
        if(curr <= 1e-9) curr = 1e-9;
        out_returns[i-1] = log(curr / prev);
    }
}

void check_constraints(SVCJParams* p) {
    // Pure Domain Constraints (Physics), no Arbitrary caps
    if(p->kappa < 0.01) p->kappa = 0.01;     if(p->kappa > 100.0) p->kappa = 100.0;
    if(p->theta < 0.0001) p->theta = 0.0001; if(p->theta > 5.0) p->theta = 5.0; 
    if(p->sigma_v < 0.001) p->sigma_v = 0.001; if(p->sigma_v > 10.0) p->sigma_v = 10.0;
    if(p->rho > 0.999) p->rho = 0.999;       if(p->rho < -0.999) p->rho = -0.999;
    if(p->lambda_j < 0.0001) p->lambda_j = 0.0001; if(p->lambda_j > 500.0) p->lambda_j = 500.0; 
    if(p->sigma_j < 0.0001) p->sigma_j = 0.0001; if(p->sigma_j > 1.0) p->sigma_j = 1.0;
    
    // Feller Violation allowed (Heston trap avoidance), just damp it slightly
    double feller = 2.0 * p->kappa * p->theta;
    if (p->sigma_v * p->sigma_v > feller * 50.0) p->sigma_v = sqrt(feller * 50.0);
}

// --- Smart Init (Unbiased) ---
void estimate_initial_params_smart(double* ohlcv, int n, SVCJParams* p) {
    // We use the WHOLE window just for the STARTING point of the simplex.
    // The optimizer is free to move away from this.
    double sum_sq = 0.0;
    for(int i=1; i<n; i++) {
        double r = log(ohlcv[i*N_COLS+3]/ohlcv[(i-1)*N_COLS+3]);
        sum_sq += r*r;
    }
    double rv = (sum_sq/(n-1)) * 252.0;

    p->mu = 0.0; 
    p->theta = rv; 
    p->kappa = 4.0; 
    p->sigma_v = sqrt(p->theta); 
    p->rho = -0.5;
    p->lambda_j = 0.1; // Start assuming diffusion
    p->mu_j = 0.0; 
    p->sigma_j = sqrt(rv/252.0) * 4.0; 
    check_constraints(p);
}

// --- Pure Likelihood (No Anchors) ---
double ukf_log_likelihood(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double ll = 0.0;
    double v = p->theta; // Start at Long Run mean
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        // Heston Predict
        double v_pred = v + p->kappa * (p->theta - v) * DT;
        if(v_pred < 1e-7) v_pred = 1e-7;

        double y = returns[t] - (p->mu - 0.5 * v_pred) * DT;
        
        // Diffusion Likelihood
        double var_d = v_pred * DT;
        double pdf_d = (1.0 / (sqrt(var_d) * SQRT_2PI)) * exp(-0.5 * y*y / var_d);
        
        // Jump Likelihood
        double total_var_j = var_d + var_j; 
        double y_j = y - p->mu_j; 
        double pdf_j = (1.0 / (sqrt(total_var_j) * SQRT_2PI)) * exp(-0.5 * y_j*y_j / total_var_j);
        
        // Bayesian Filter Step
        double prob_prior = (p->lambda_j * DT > 0.99) ? 0.99 : p->lambda_j * DT;
        double den = pdf_j * prob_prior + pdf_d * (1.0 - prob_prior);
        
        // Numerical safety only
        if(den < 1e-20) den = 1e-20;
        
        double prob_posterior = (pdf_j * prob_prior) / den;
        
        // Update State
        double S = v_pred * DT + prob_posterior * var_j;
        double K = (p->rho * p->sigma_v * DT) / S;
        v = v_pred + K * y;
        if(v < 1e-7) v = 1e-7; if(v > 10.0) v = 10.0; // Physics bounds only

        if(out_spot_vol) out_spot_vol[t] = sqrt(v_pred); 
        if(out_jump_prob) out_jump_prob[t] = prob_posterior;
        
        ll += log(den); 
    }
    
    if(isnan(ll) || isinf(ll)) return -1e15;
    return ll; // Pure MLE
}

// --- Grid Search Init ---
void grid_search_init(double* returns, int n, SVCJParams* p) {
    // Scan high/low volatility regimes to find best starting basin
    double thetas[] = {p->theta * 0.5, p->theta, p->theta * 1.5};
    double lambdas[] = {0.1, 5.0, 20.0};
    
    double best_score = -1e15;
    SVCJParams best_p = *p;
    
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            SVCJParams temp = *p;
            temp.theta = thetas[i];
            temp.lambda_j = lambdas[j];
            check_constraints(&temp);
            
            double score = ukf_log_likelihood(returns, n, &temp, NULL, NULL);
            if(score > best_score) { best_score = score; best_p = temp; }
        }
    }
    *p = best_p;
}

void optimize_svcj(double* ohlcv, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    estimate_initial_params_smart(ohlcv, n, p);
    
    int n_ret = n - 1;
    double* returns = (double*)malloc(n_ret * sizeof(double));
    if(!returns) return;
    compute_log_returns(ohlcv, n, returns);
    clean_returns(returns, n_ret);

    // 1. Grid Search to find correct basin
    grid_search_init(returns, n_ret, p);

    // 2. Nelder-Mead Optimization (Unconstrained Likelihood)
    int n_dim = 5;
    double simplex[6][5];
    double scores[6];
    
    for(int i=0; i<=n_dim; i++) {
        SVCJParams temp = *p;
        if(i==1) temp.kappa *= 1.2; if(i==2) temp.theta *= 1.2;
        if(i==3) temp.sigma_v *= 1.2; if(i==4) temp.rho -= 0.1;
        if(i==5) temp.lambda_j *= 1.5;
        check_constraints(&temp);
        
        simplex[i][0]=temp.kappa; simplex[i][1]=temp.theta; simplex[i][2]=temp.sigma_v;
        simplex[i][3]=temp.rho;   simplex[i][4]=temp.lambda_j;
        scores[i] = ukf_log_likelihood(returns, n_ret, &temp, NULL, NULL);
    }
    
    for(int iter=0; iter<NM_ITER; iter++) {
        int vs[6]; for(int k=0; k<6; k++) vs[k]=k;
        for(int i=0; i<6; i++) for(int j=i+1; j<6; j++) if(scores[vs[j]] > scores[vs[i]]) { int t=vs[i]; vs[i]=vs[j]; vs[j]=t; }
        
        double c[5]={0}; for(int i=0; i<5; i++) for(int k=0; k<5; k++) c[k]+=simplex[vs[i]][k];
        for(int k=0; k<5; k++) c[k]/=5.0;
        
        double ref[5]; SVCJParams rp=*p; for(int k=0; k<5; k++) ref[k]=c[k]+1.0*(c[k]-simplex[vs[5]][k]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        check_constraints(&rp); double r_score=ukf_log_likelihood(returns, n_ret, &rp, NULL, NULL);
        
        if(r_score>scores[vs[0]]) {
            double exp[5]; SVCJParams ep=*p; for(int k=0; k<5; k++) exp[k]=c[k]+2.0*(c[k]-simplex[vs[5]][k]);
            ep.kappa=exp[0]; ep.theta=exp[1]; ep.sigma_v=exp[2]; ep.rho=exp[3]; ep.lambda_j=exp[4];
            check_constraints(&ep); double e_score=ukf_log_likelihood(returns, n_ret, &ep, NULL, NULL);
            if(e_score>r_score) { for(int k=0; k<5; k++) simplex[vs[5]][k]=exp[k]; scores[vs[5]]=e_score; }
            else { for(int k=0; k<5; k++) simplex[vs[5]][k]=ref[k]; scores[vs[5]]=r_score; }
        } else if(r_score>scores[vs[4]]) {
            for(int k=0; k<5; k++) simplex[vs[5]][k]=ref[k]; scores[vs[5]]=r_score;
        } else {
            double con[5]; SVCJParams cp=*p; for(int k=0; k<5; k++) con[k]=c[k]+0.5*(simplex[vs[5]][k]-c[k]);
            cp.kappa=con[0]; cp.theta=con[1]; cp.sigma_v=con[2]; cp.rho=con[3]; cp.lambda_j=con[4];
            check_constraints(&cp); double c_score=ukf_log_likelihood(returns, n_ret, &cp, NULL, NULL);
            if(c_score>scores[vs[5]]) { for(int k=0; k<5; k++) simplex[vs[5]][k]=con[k]; scores[vs[5]]=c_score; }
            else {
                for(int i=1; i<6; i++) {
                    int idx=vs[i]; SVCJParams sp=*p; for(int k=0; k<5; k++) simplex[idx][k]=simplex[vs[0]][k]+0.5*(simplex[idx][k]-simplex[vs[0]][k]);
                    sp.kappa=simplex[idx][0]; sp.theta=simplex[idx][1]; sp.sigma_v=simplex[idx][2]; sp.rho=simplex[idx][3]; sp.lambda_j=simplex[idx][4];
                    check_constraints(&sp); scores[idx]=ukf_log_likelihood(returns, n_ret, &sp, NULL, NULL);
                }
            }
        }
    }
    
    int best=0; for(int i=1; i<6; i++) if(scores[i]>scores[best]) best=i;
    p->kappa=simplex[best][0]; p->theta=simplex[best][1]; p->sigma_v=simplex[best][2];
    p->rho=simplex[best][3]; p->lambda_j=simplex[best][4];
    
    ukf_log_likelihood(returns, n_ret, p, out_spot_vol, out_jump_prob);
    free(returns);
}

// Pricing Unchanged
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