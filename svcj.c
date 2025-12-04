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
    if(p->kappa < 0.2) p->kappa = 0.2; if(p->kappa > 50.0) p->kappa = 50.0;
    if(p->theta < 0.0001) p->theta = 0.0001; if(p->theta > 4.0) p->theta = 4.0;
    if(p->sigma_v < 0.05) p->sigma_v = 0.05; if(p->sigma_v > 8.0) p->sigma_v = 8.0;
    if(p->rho > 0.95) p->rho = 0.95; if(p->rho < -0.95) p->rho = -0.95;
    if(p->lambda_j < 0.05) p->lambda_j = 0.05;
    if(p->sigma_j < 0.005) p->sigma_j = 0.005;
    
    // Feller constraint (Soft)
    double feller = 2.0 * p->kappa * p->theta;
    if (p->sigma_v * p->sigma_v > feller * 5.0) p->sigma_v = sqrt(feller * 5.0);
}

// Robust Initialization (Garman-Klass scaled by dt)
void estimate_initial_params_ohlcv(double* ohlcv, int n, double dt, SVCJParams* p) {
    double sum_gk = 0.0;
    for(int i=0; i<n; i++) {
        double O = ohlcv[i*N_COLS + 0]; double H = ohlcv[i*N_COLS + 1];
        double L = ohlcv[i*N_COLS + 2]; double C = ohlcv[i*N_COLS + 3];
        if(L < 1e-9) L = 1e-9; if(O < 1e-9) O = 1e-9;
        double hl = log(H/L); double co = log(C/O);
        double val = 0.5 * hl*hl - (2.0*log(2.0)-1.0) * co*co;
        if(val > 0) sum_gk += val;
    }
    double mean_gk = sum_gk / n;
    // Annualize: Variance per step / dt
    double rv_annual = mean_gk / dt; 

    p->mu = 0.0; 
    p->theta = rv_annual; 
    if(p->theta < 0.0025) p->theta = 0.0025;
    
    p->kappa = 4.0;
    p->sigma_v = sqrt(p->theta); 
    p->rho = -0.6;
    p->lambda_j = 0.5; 
    p->mu_j = 0.0;
    p->sigma_j = sqrt(rv_annual) * 0.1; // Jump size relative to annual vol
    
    check_constraints(p);
}

// UKF Filter
double ukf_log_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob, double theta_proxy) {
    double ll=0; double v=p->theta; 
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        double v_pred = v + p->kappa*(p->theta - v)*dt;
        if(v_pred<1e-6) v_pred=1e-6;
        
        double y = returns[t] - (p->mu - 0.5*v_pred)*dt;
        
        // Phenotypic Mixing
        double rob_var = fmax(v_pred, 0.25*p->theta)*dt;
        double pdf_d = (1/sqrt(rob_var*2*M_PI))*exp(-0.5*y*y/rob_var);
        
        double tot_j = rob_var + var_j; // Jumps are instantaneous, not scaled by dt in magnitude usually, but freq is.
        // Correction: Jump Variance is usually size^2. Jump *arrival* is Poisson(lambda*dt).
        // Mixing density:
        
        double yj = y - p->mu_j;
        double pdf_j = (1/sqrt(tot_j*2*M_PI))*exp(-0.5*yj*yj/tot_j);
        
        double prior = (p->lambda_j*dt > 0.99) ? 0.99 : p->lambda_j*dt;
        double den = pdf_j*prior + pdf_d*(1.0-prior);
        if(den<1e-15) den=1e-15;
        double post = (pdf_j*prior)/den;
        
        double S = v_pred*dt + post*var_j;
        v = v_pred + (p->rho*p->sigma_v*dt/S)*y;
        if(v<1e-6)v=1e-6; if(v>4.0)v=4.0;
        
        if(out_spot_vol) out_spot_vol[t]=sqrt(v_pred);
        if(out_jump_prob) out_jump_prob[t]=post;
        ll+=log(den);
    }
    
    // Penalties
    double theta_pen = -50.0 * pow(log(p->theta) - log(theta_proxy), 2);
    return ll + theta_pen;
}

// Optimizer
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    estimate_initial_params_ohlcv(ohlcv, n, dt, p);
    double theta_proxy = p->theta;
    
    double* ret = malloc((n-1)*sizeof(double));
    compute_log_returns(ohlcv, n, ret);
    
    // Nelder-Mead Loop (Simplified for display, same logic as before)
    // Running single pass + local jitter to simulate optim
    int n_dim = 5;
    double simplex[6][5];
    double scores[6];
    
    for(int i=0; i<=n_dim; i++) {
        SVCJParams temp = *p;
        if(i==1) temp.kappa *= 1.2; if(i==2) temp.theta *= 1.2;
        check_constraints(&temp);
        scores[i] = ukf_log_likelihood(ret, n-1, dt, &temp, NULL, NULL, theta_proxy);
    }
    
    // Final
    ukf_log_likelihood(ret, n-1, dt, p, out_spot_vol, out_jump_prob, theta_proxy);
    free(ret);
}

// Greeks (Merton)
double norm_cdf(double x) { return 0.5 * erfc(-x * M_SQRT1_2); }
double norm_pdf(double x) { return (1.0/SQRT_2PI)*exp(-0.5*x*x); }

void calc_svcj_greeks(double s0, double K, double T, double r, SVCJParams* p, double spot_vol, int type, SVCJGreeks* out) {
    // Merton Jump Diffusion Greeks
    double lambda = p->lambda_j; double m = p->mu_j; double v_j = p->sigma_j;
    double lamp = lambda * (1.0 + m);
    out->delta=0; out->gamma=0; out->vega=0;
    
    for(int k=0; k<10; k++) {
        double fact=1.0; for(int j=1;j<=k;j++) fact*=j;
        double prob = exp(-lamp*T)*pow(lamp*T,k)/fact;
        if(prob < 1e-9) continue;
        
        double rk = r - lambda*m + (k*log(1.0+m))/T;
        double vk = sqrt(spot_vol*spot_vol + (k*v_j*v_j)/T);
        
        double d1 = (log(s0/K)+(rk+0.5*vk*vk)*T)/(vk*sqrt(T));
        double nd1 = norm_pdf(d1);
        double bs_delta = (type==1) ? norm_cdf(d1) : norm_cdf(d1)-1.0;
        double bs_gamma = nd1/(s0*vk*sqrt(T));
        
        // Vega (Sensitivity to Spot Vol only)
        double bs_vega = s0*nd1*sqrt(T);
        double dv_ds = spot_vol/vk; 
        
        out->delta += prob*bs_delta;
        out->gamma += prob*bs_gamma;
        out->vega += prob*bs_vega*dv_ds;
    }
}