#include "svcj.h"
#include <float.h>

void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) {
    for(int i=1; i<n_rows; i++) {
        double prev = ohlcv[(i-1)*N_COLS + 3];
        double curr = ohlcv[i*N_COLS + 3];
        if(prev < 1e-9) prev = 1e-9;
        out_returns[i-1] = log(curr / prev);
    }
}

void check_constraints(SVCJParams* p) {
    // Soft Constraints: Wide bounds to allow fitting extreme data
    if(p->kappa < 0.01) p->kappa = 0.01;   if(p->kappa > 100.0) p->kappa = 100.0;
    if(p->theta < 1e-6) p->theta = 1e-6;   if(p->theta > 50.0) p->theta = 50.0;
    if(p->sigma_v < 0.01) p->sigma_v = 0.01; if(p->sigma_v > 50.0) p->sigma_v = 50.0;
    if(p->rho > 0.999) p->rho = 0.999;     if(p->rho < -0.999) p->rho = -0.999;
    if(p->lambda_j < 0.001) p->lambda_j = 0.001; if(p->lambda_j > 3000.0) p->lambda_j = 3000.0;
    if(p->sigma_j < 0.0001) p->sigma_j = 0.0001;
    
    // Feller Violation allowed but capped at 20x to prevent NaN
    double feller = 2.0 * p->kappa * p->theta;
    if (p->sigma_v * p->sigma_v > feller * 20.0) p->sigma_v = sqrt(feller * 20.0);
}

void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p) {
    double sum_gk = 0.0;
    for(int i=0; i<n; i++) {
        double O = ohlcv[i*N_COLS]; double H = ohlcv[i*N_COLS+1];
        double L = ohlcv[i*N_COLS+2]; double C = ohlcv[i*N_COLS+3];
        if(L<1e-9)L=1e-9; if(O<1e-9)O=1e-9;
        double hl = log(H/L); double co = log(C/O);
        double val = 0.5*hl*hl - (2.0*log(2.0)-1.0)*co*co;
        if(val>0) sum_gk += val;
    }
    double rv = (sum_gk / n) / dt;
    
    p->mu = 0.0; p->theta = rv;
    p->kappa = 2.0; p->sigma_v = sqrt(rv); p->rho = -0.5;
    p->lambda_j = 0.5; p->mu_j = 0.0; p->sigma_j = sqrt(rv); 
    check_constraints(p);
}

// 1. Pure Likelihood Function (No Penalties)
// Used for the Statistical Test and Final State Output
double ukf_pure_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double ll = 0.0; 
    double v = p->theta;
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        double v_pred = v + p->kappa*(p->theta - v)*dt;
        if(v_pred < 1e-9) v_pred = 1e-9;

        double drift = (p->mu - 0.5*v_pred);
        double y = returns[t] - drift*dt;

        // Phenotypic Mixing logic kept for numerical stability of the Filter
        double rob_var = (v_pred < 1e-9) ? 1e-9 : v_pred;
        rob_var *= dt; // variance over step dt
        
        double pdf_d = (1.0/sqrt(rob_var*2*M_PI)) * exp(-0.5*y*y/rob_var);
        
        double tot_j = rob_var + var_j; // Jumps are instantaneous
        double yj = y - p->mu_j;
        double pdf_j = (1.0/sqrt(tot_j*2*M_PI)) * exp(-0.5*yj*yj/tot_j);
        
        double prior = p->lambda_j * dt;
        if(prior > 0.999) prior = 0.999;
        
        double den = pdf_j*prior + pdf_d*(1.0-prior);
        if(den < 1e-25) den = 1e-25; // Floor to prevent log(0)
        double post = (pdf_j*prior)/den;
        
        double S = v_pred*dt + post*var_j;
        v = v_pred + (p->rho*p->sigma_v*dt/S)*y;
        if(v<1e-9)v=1e-9; if(v>50.0)v=50.0;
        
        if(out_spot_vol) out_spot_vol[t]=sqrt(v_pred);
        if(out_jump_prob) out_jump_prob[t]=post;
        
        ll += log(den);
    }
    return ll;
}

// 2. Objective Function (Likelihood + Soft Constraints)
// Used ONLY by the Optimizer to guide convergence
double objective_function(double* returns, int n, double dt, SVCJParams* p) {
    double ll = ukf_pure_likelihood(returns, n, dt, p, NULL, NULL);
    
    // Minimal regularization to keep parameters in physical reality
    // Penalize negative Theta or crazy Sigma_V effectively via soft constraints
    // Here we assume check_constraints handles hard limits, so we add weak priors
    // to prefer simpler models (Parsimony)
    
    // Weak penalty on very high Vol of Vol (Occam's Razor)
    double penalty = -0.1 * (p->sigma_v * p->sigma_v); 
    
    return ll + penalty;
}

void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    estimate_initial_params(ohlcv, n, dt, p);
    
    double* ret = malloc((n-1)*sizeof(double));
    if(!ret) return;
    compute_log_returns(ohlcv, n, ret);
    
    int n_dim=5; double simplex[6][5]; double scores[6];
    
    // Init
    for(int i=0; i<=n_dim; i++) {
        SVCJParams t = *p;
        if(i==1) t.kappa*=1.3; if(i==2) t.theta*=1.3; if(i==3) t.sigma_v*=1.3;
        if(i==4) t.rho+=0.2;   if(i==5) t.lambda_j*=1.5;
        check_constraints(&t);
        simplex[i][0]=t.kappa; simplex[i][1]=t.theta; simplex[i][2]=t.sigma_v;
        simplex[i][3]=t.rho;   simplex[i][4]=t.lambda_j;
        scores[i] = objective_function(ret, n-1, dt, &t);
    }
    
    // Loop
    for(int k=0; k<NM_ITER; k++) {
        int vs[6]; for(int j=0; j<6; j++) vs[j]=j;
        for(int i=0; i<6; i++) {
             for(int j=i+1; j<6; j++) { if(scores[vs[j]] > scores[vs[i]]) { int tmp=vs[i]; vs[i]=vs[j]; vs[j]=tmp; } }
        }
        double c[5]={0};
        for(int i=0; i<5; i++) { for(int d=0; d<5; d++) c[d]+=simplex[vs[i]][d]; }
        for(int d=0; d<5; d++) c[d]/=5.0;
        
        double ref[5]; SVCJParams rp = *p;
        for(int d=0; d<5; d++) ref[d] = c[d] + 1.0*(c[d]-simplex[vs[5]][d]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        check_constraints(&rp);
        double r_score = objective_function(ret, n-1, dt, &rp);
        
        if(r_score > scores[vs[0]]) {
             for(int d=0; d<5; d++) simplex[vs[5]][d] = ref[d]; scores[vs[5]] = r_score;
        } else {
             double con[5]; SVCJParams cp = *p;
             for(int d=0; d<5; d++) con[d] = c[d] + 0.5*(simplex[vs[5]][d]-c[d]);
             cp.kappa=con[0]; cp.theta=con[1]; cp.sigma_v=con[2]; cp.rho=con[3]; cp.lambda_j=con[4];
             check_constraints(&cp);
             double c_score = objective_function(ret, n-1, dt, &cp);
             if(c_score > scores[vs[5]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = con[d]; scores[vs[5]] = c_score; }
        }
    }
    
    int best=0; for(int i=1; i<6; i++) if(scores[i]>scores[best]) best=i;
    p->kappa=simplex[best][0]; p->theta=simplex[best][1]; p->sigma_v=simplex[best][2];
    p->rho=simplex[best][3];   p->lambda_j=simplex[best][4];
    
    // Output Pure Likelihood State
    ukf_pure_likelihood(ret, n-1, dt, p, out_spot_vol, out_jump_prob);
    free(ret);
}

double chi2_prob(double x, int k) {
    if(x <= 0) return 1.0;
    double s = 2.0/9.0/k;
    double z = (pow(x/k, 1.0/3.0) - (1.0 - s)) / sqrt(s);
    return 0.5 * erfc(z * M_SQRT1_2);
}

void perform_likelihood_test(double* ohlcv, int len_long, int len_short, double dt, RegimeStats* out) {
    SVCJParams p_long, p_short;
    
    // 1. Fit Long (Null Hypothesis)
    optimize_svcj(ohlcv, len_long, dt, &p_long, NULL, NULL);
    
    // 2. Fit Short (Alternative Hypothesis)
    int offset = len_long - len_short;
    optimize_svcj(ohlcv + offset*N_COLS, len_short, dt, &p_short, NULL, NULL);
    
    // 3. Calc Pure Likelihoods on Short Data
    double* ret_short = malloc((len_short-1)*sizeof(double));
    compute_log_returns(ohlcv + offset*N_COLS, len_short, ret_short);
    
    out->ll_null = ukf_pure_likelihood(ret_short, len_short-1, dt, &p_long, NULL, NULL);
    out->ll_alt = ukf_pure_likelihood(ret_short, len_short-1, dt, &p_short, NULL, NULL);
    
    free(ret_short);
    
    out->statistic = 2.0 * (out->ll_alt - out->ll_null);
    if(out->statistic < 0) out->statistic = 0;
    
    out->p_value = chi2_prob(out->statistic, 5); // 5 free parameters
    out->significant = (out->p_value < 0.05) ? 1 : 0;
}

// Greeks
double norm_cdf(double x) { return 0.5 * erfc(-x * M_SQRT1_2); }
void calc_greeks(double s0, double K, double T, double r, SVCJParams* p, double spot_vol, int type, SVCJGreeks* out) {
    double lambda = p->lambda_j; double lamp = lambda * (1.0 + p->mu_j);
    out->delta = 0; out->gamma = 0; out->vega = 0;
    for(int k=0; k<15; k++) {
        double fact=1.0; for(int j=1;j<=k;j++) fact*=j;
        double prob = exp(-lamp*T)*pow(lamp*T,k)/fact;
        if(prob < 1e-9) continue;
        double rk = r - lambda*p->mu_j + (k*log(1.0+p->mu_j))/T;
        double vk = sqrt(spot_vol*spot_vol + (k*p->sigma_j*p->sigma_j)/T);
        double d1 = (log(s0/K) + (rk+0.5*vk*vk)*T)/(vk*sqrt(T));
        double d2 = d1 - vk*sqrt(T);
        double delta = (type==1) ? norm_cdf(d1) : norm_cdf(d1)-1.0;
        out->delta += prob*delta;
        double nd1 = (1.0/SQRT_2PI)*exp(-0.5*d1*d1);
        out->gamma += prob * (nd1 / (s0*vk*sqrt(T)));
        double dv_ds = spot_vol / vk;
        out->vega += prob * (s0 * nd1 * sqrt(T)) * dv_ds;
    }
}