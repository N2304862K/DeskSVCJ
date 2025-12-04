#include "svcj.h"
#include <float.h>

// Helper: Calculate Log Returns from Close column (Index 3)
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) {
    for(int i=1; i<n_rows; i++) {
        double prev = ohlcv[(i-1)*N_COLS + 3];
        double curr = ohlcv[i*N_COLS + 3];
        if(prev < 1e-9) prev = 1e-9;
        out_returns[i-1] = log(curr / prev);
    }
}

// Constraints to prevent numerical explosions
void check_constraints(SVCJParams* p) {
    if(p->kappa < 0.1) p->kappa = 0.1; if(p->kappa > 50.0) p->kappa = 50.0;
    if(p->theta < 1e-5) p->theta = 1e-5; if(p->theta > 5.0) p->theta = 5.0;
    if(p->sigma_v < 0.05) p->sigma_v = 0.05; if(p->sigma_v > 8.0) p->sigma_v = 8.0;
    if(p->rho > 0.99) p->rho = 0.99; if(p->rho < -0.99) p->rho = -0.99;
    
    // Jump constraints
    if(p->lambda_j < 0.01) p->lambda_j = 0.01; if(p->lambda_j > 500.0) p->lambda_j = 500.0;
    if(p->sigma_j < 0.001) p->sigma_j = 0.001;
    
    // Feller Condition (Soft)
    double feller = 2.0 * p->kappa * p->theta;
    if (p->sigma_v * p->sigma_v > feller * 5.0) p->sigma_v = sqrt(feller * 5.0);
}

// Standalone Initialization (Garman-Klass)
// Uses OHLC info to estimate variance without needing Option Data
void estimate_initial_params_ohlcv(double* ohlcv, int n, double dt, SVCJParams* p) {
    double sum_gk = 0.0;
    for(int i=0; i<n; i++) {
        double O = ohlcv[i*N_COLS+0]; double H = ohlcv[i*N_COLS+1];
        double L = ohlcv[i*N_COLS+2]; double C = ohlcv[i*N_COLS+3];
        if(L<1e-9)L=1e-9; if(O<1e-9)O=1e-9;
        
        double hl = log(H/L); double co = log(C/O);
        double val = 0.5*hl*hl - (2.0*log(2.0)-1.0)*co*co;
        if(val>0) sum_gk += val;
    }
    double mean_gk = sum_gk / n;
    double rv_annual = mean_gk / dt; // Normalize based on provided dt

    p->mu = 0.0;
    p->theta = rv_annual;
    p->kappa = 4.0; 
    p->sigma_v = sqrt(p->theta); 
    p->rho = -0.6;
    p->lambda_j = 0.5; 
    p->mu_j = 0.0;
    p->sigma_j = sqrt(rv_annual) * 2.0; 
    check_constraints(p);
}

// UKF Filter with Phenotypic Mixing
double ukf_log_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob, double theta_proxy) {
    double ll=0; double v=p->theta;
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        // 1. Prediction
        double v_pred = v + p->kappa*(p->theta - v)*dt;
        if(v_pred < 1e-6) v_pred = 1e-6;

        // 2. Innovation
        double drift = (p->mu - 0.5*v_pred);
        double y = returns[t] - drift*dt;

        // 3. Phenotypic Mixing (Prevents Diffusion Collapse)
        // We ensure variance doesn't drop below 25% of Long-Run Theta
        double rob_var = fmax(v_pred, 0.25*p->theta)*dt;
        
        double pdf_d = (1.0/sqrt(rob_var*2*M_PI)) * exp(-0.5*y*y/rob_var);
        
        double tot_j = rob_var + var_j; 
        double yj = y - p->mu_j;
        double pdf_j = (1.0/sqrt(tot_j*2*M_PI)) * exp(-0.5*yj*yj/tot_j);
        
        double prior = p->lambda_j * dt;
        if(prior > 0.9) prior = 0.9;
        
        double den = pdf_j*prior + pdf_d*(1.0-prior);
        if(den < 1e-15) den = 1e-15;
        double post = (pdf_j*prior)/den;
        
        // 4. Update
        double S = v_pred*dt + post*var_j;
        v = v_pred + (p->rho*p->sigma_v*dt/S)*y;
        if(v<1e-6)v=1e-6; if(v>4.0)v=4.0;
        
        if(out_spot_vol) out_spot_vol[t]=sqrt(v_pred);
        if(out_jump_prob) out_jump_prob[t]=post;
        ll += log(den);
    }
    
    // Penalize deviation from Ground Truth Theta (Garman-Klass)
    double theta_pen = -50.0 * pow(log(p->theta) - log(theta_proxy), 2);
    return ll + theta_pen;
}

// Optimizer (Nelder-Mead)
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    estimate_initial_params_ohlcv(ohlcv, n, dt, p);
    double theta_proxy = p->theta;
    
    double* ret = malloc((n-1)*sizeof(double));
    if(!ret) return;
    compute_log_returns(ohlcv, n, ret);
    
    int n_dim=5; double simplex[6][5]; double scores[6];
    
    // 1. Init Simplex
    for(int i=0; i<=n_dim; i++) {
        SVCJParams t = *p;
        if(i==1) t.kappa*=1.2; if(i==2) t.theta*=1.2; if(i==3) t.sigma_v*=1.2;
        if(i==4) t.rho+=0.2;   if(i==5) t.lambda_j*=1.5;
        check_constraints(&t);
        simplex[i][0]=t.kappa; simplex[i][1]=t.theta; simplex[i][2]=t.sigma_v;
        simplex[i][3]=t.rho;   simplex[i][4]=t.lambda_j;
        scores[i] = ukf_log_likelihood(ret, n-1, dt, &t, NULL, NULL, theta_proxy);
    }
    
    // 2. Optimization Loop
    for(int k=0; k<NM_ITER; k++) {
        // Sort
        int vs[6]; for(int j=0; j<6; j++) vs[j]=j;
        for(int i=0; i<6; i++) {
             for(int j=i+1; j<6; j++) { if(scores[vs[j]] > scores[vs[i]]) { int tmp=vs[i]; vs[i]=vs[j]; vs[j]=tmp; } }
        }
        
        // Centroid
        double c[5]={0};
        for(int i=0; i<5; i++) { for(int d=0; d<5; d++) c[d]+=simplex[vs[i]][d]; }
        for(int d=0; d<5; d++) c[d]/=5.0;
        
        // Reflect
        double ref[5]; SVCJParams rp = *p;
        for(int d=0; d<5; d++) ref[d] = c[d] + 1.0*(c[d]-simplex[vs[5]][d]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        check_constraints(&rp);
        double r_score = ukf_log_likelihood(ret, n-1, dt, &rp, NULL, NULL, theta_proxy);
        
        if(r_score > scores[vs[0]]) {
             for(int d=0; d<5; d++) simplex[vs[5]][d] = ref[d]; scores[vs[5]] = r_score;
        } else {
             // Contract
             double con[5]; SVCJParams cp = *p;
             for(int d=0; d<5; d++) con[d] = c[d] + 0.5*(simplex[vs[5]][d]-c[d]);
             cp.kappa=con[0]; cp.theta=con[1]; cp.sigma_v=con[2]; cp.rho=con[3]; cp.lambda_j=con[4];
             check_constraints(&cp);
             double c_score = ukf_log_likelihood(ret, n-1, dt, &cp, NULL, NULL, theta_proxy);
             if(c_score > scores[vs[5]]) {
                 for(int d=0; d<5; d++) simplex[vs[5]][d] = con[d]; scores[vs[5]] = c_score;
             }
        }
    }
    
    int best=0; for(int i=1; i<6; i++) if(scores[i]>scores[best]) best=i;
    p->kappa=simplex[best][0]; p->theta=simplex[best][1]; p->sigma_v=simplex[best][2];
    p->rho=simplex[best][3];   p->lambda_j=simplex[best][4];
    
    ukf_log_likelihood(ret, n-1, dt, p, out_spot_vol, out_jump_prob, theta_proxy);
    free(ret);
}

// Analytic Greeks (Merton Expansion)
double norm_pdf(double x) { return (1.0/SQRT_2PI)*exp(-0.5*x*x); }
double norm_cdf(double x) { return 0.5 * erfc(-x * M_SQRT1_2); }

void calc_svcj_greeks(double s0, double K, double T, double r, SVCJParams* p, double spot_vol, int type, SVCJGreeks* out) {
    double lambda = p->lambda_j; double m = p->mu_j; double v_j = p->sigma_j;
    double lamp = lambda * (1.0 + m);
    
    out->delta = 0; out->gamma = 0; out->vega = 0;
    
    for(int k=0; k<12; k++) {
        double fact=1.0; for(int j=1;j<=k;j++) fact*=j;
        double prob = exp(-lamp*T)*pow(lamp*T,k)/fact;
        if(prob < 1e-9 && k>2) break;
        
        double rk = r - lambda*m + (k*log(1.0+m))/T;
        double vk = sqrt(spot_vol*spot_vol + (k*v_j*v_j)/T);
        if(vk<1e-5) vk=1e-5;
        
        double d1 = (log(s0/K) + (rk+0.5*vk*vk)*T)/(vk*sqrt(T));
        double nd1 = norm_pdf(d1);
        
        double bs_delta = (type==1) ? norm_cdf(d1) : norm_cdf(d1)-1.0;
        double bs_gamma = nd1 / (s0*vk*sqrt(T));
        double bs_vega  = s0 * nd1 * sqrt(T);
        double dv_ds = spot_vol / vk; // Chain rule
        
        out->delta += prob*bs_delta;
        out->gamma += prob*bs_gamma;
        out->vega  += prob*bs_vega*dv_ds;
    }
}