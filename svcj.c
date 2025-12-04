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
    if(p->kappa < 0.1) p->kappa = 0.1; if(p->kappa > 50.0) p->kappa = 50.0;
    if(p->theta < 1e-6) p->theta = 1e-6; if(p->theta > 10.0) p->theta = 10.0;
    if(p->sigma_v < 0.01) p->sigma_v = 0.01; if(p->sigma_v > 10.0) p->sigma_v = 10.0;
    if(p->rho > 0.99) p->rho = 0.99; if(p->rho < -0.99) p->rho = -0.99;
    if(p->lambda_j < 0.01) p->lambda_j = 0.01; if(p->lambda_j > 1000.0) p->lambda_j = 1000.0;
    if(p->sigma_j < 0.001) p->sigma_j = 0.001;
    
    double feller = 2.0 * p->kappa * p->theta;
    if (p->sigma_v * p->sigma_v > feller * 6.0) p->sigma_v = sqrt(feller * 6.0);
}

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
    double rv_annual = mean_gk / dt; // Normalize

    p->mu = 0.0; p->theta = rv_annual;
    p->kappa = 4.0; p->sigma_v = sqrt(p->theta); p->rho = -0.6;
    p->lambda_j = 0.5; p->mu_j = 0.0; p->sigma_j = sqrt(rv_annual) * 2.0; 
    check_constraints(p);
}

double ukf_log_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob, double theta_proxy) {
    double ll=0; double v=p->theta;
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        double v_pred = v + p->kappa*(p->theta - v)*dt;
        if(v_pred < 1e-7) v_pred = 1e-7;

        double drift = (p->mu - 0.5*v_pred);
        double y = returns[t] - drift*dt;

        double rob_var = fmax(v_pred, 0.25*p->theta)*dt;
        double pdf_d = (1.0/sqrt(rob_var*2*M_PI)) * exp(-0.5*y*y/rob_var);
        
        double tot_j = rob_var + var_j; 
        double yj = y - p->mu_j;
        double pdf_j = (1.0/sqrt(tot_j*2*M_PI)) * exp(-0.5*yj*yj/tot_j);
        
        double prior = p->lambda_j * dt;
        if(prior > 0.99) prior = 0.99;
        
        double den = pdf_j*prior + pdf_d*(1.0-prior);
        if(den < 1e-18) den = 1e-18;
        double post = (pdf_j*prior)/den;
        
        double S = v_pred*dt + post*var_j;
        v = v_pred + (p->rho*p->sigma_v*dt/S)*y;
        if(v<1e-7)v=1e-7; if(v>5.0)v=5.0;
        
        if(out_spot_vol) out_spot_vol[t]=sqrt(v_pred);
        if(out_jump_prob) out_jump_prob[t]=post;
        ll += log(den);
    }
    double theta_pen = -50.0 * pow(log(p->theta) - log(theta_proxy), 2);
    return ll + theta_pen;
}

double raw_likelihood(double* returns, int n, double dt, SVCJParams* p) {
    double ll=0; double v=p->theta;
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    for(int t=0; t<n; t++) {
        double v_pred = v + p->kappa*(p->theta - v)*dt;
        if(v_pred<1e-7)v_pred=1e-7;
        double y = returns[t] - (p->mu - 0.5*v_pred)*dt;
        double rob_var = fmax(v_pred, 0.25*p->theta)*dt;
        double pdf_d = (1.0/sqrt(rob_var*2*M_PI))*exp(-0.5*y*y/rob_var);
        double tot_j = rob_var + var_j; 
        double yj = y - p->mu_j;
        double pdf_j = (1.0/sqrt(tot_j*2*M_PI))*exp(-0.5*yj*yj/tot_j);
        double prior = (p->lambda_j*dt > 0.99) ? 0.99 : p->lambda_j*dt;
        double den = pdf_j*prior + pdf_d*(1.0-prior);
        if(den<1e-18)den=1e-18;
        double post = (pdf_j*prior)/den;
        double S = v_pred*dt + post*var_j;
        v = v_pred + (p->rho*p->sigma_v*dt/S)*y;
        if(v<1e-7)v=1e-7; if(v>5.0)v=5.0;
        ll+=log(den);
    }
    return ll;
}

void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    estimate_initial_params_ohlcv(ohlcv, n, dt, p);
    double theta_proxy = p->theta;
    double* ret = malloc((n-1)*sizeof(double));
    compute_log_returns(ohlcv, n, ret);
    
    int n_dim=5; double simplex[6][5]; double scores[6];
    for(int i=0; i<=n_dim; i++) {
        SVCJParams t = *p;
        if(i==1) t.kappa*=1.2; if(i==2) t.theta*=1.2; if(i==3) t.sigma_v*=1.2;
        if(i==4) t.rho+=0.2;   if(i==5) t.lambda_j*=1.5;
        check_constraints(&t);
        simplex[i][0]=t.kappa; simplex[i][1]=t.theta; simplex[i][2]=t.sigma_v;
        simplex[i][3]=t.rho;   simplex[i][4]=t.lambda_j;
        scores[i] = ukf_log_likelihood(ret, n-1, dt, &t, NULL, NULL, theta_proxy);
    }
    
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
        double r_score = ukf_log_likelihood(ret, n-1, dt, &rp, NULL, NULL, theta_proxy);
        if(r_score > scores[vs[0]]) {
             for(int d=0; d<5; d++) simplex[vs[5]][d] = ref[d]; scores[vs[5]] = r_score;
        } else {
             double con[5]; SVCJParams cp = *p;
             for(int d=0; d<5; d++) con[d] = c[d] + 0.5*(simplex[vs[5]][d]-c[d]);
             cp.kappa=con[0]; cp.theta=con[1]; cp.sigma_v=con[2]; cp.rho=con[3]; cp.lambda_j=con[4];
             check_constraints(&cp);
             double c_score = ukf_log_likelihood(ret, n-1, dt, &cp, NULL, NULL, theta_proxy);
             if(c_score > scores[vs[5]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = con[d]; scores[vs[5]] = c_score; }
        }
    }
    int best=0; for(int i=1; i<6; i++) if(scores[i]>scores[best]) best=i;
    p->kappa=simplex[best][0]; p->theta=simplex[best][1]; p->sigma_v=simplex[best][2];
    p->rho=simplex[best][3];   p->lambda_j=simplex[best][4];
    
    if(out_spot_vol) ukf_log_likelihood(ret, n-1, dt, p, out_spot_vol, out_jump_prob, theta_proxy);
    free(ret);
}

// Chi-Square Prob for LRT
double chi2_p_value(double x, int k) {
    if(x <= 0) return 1.0;
    double s = 2.0/9.0/k;
    double z = (pow(x/k, 1.0/3.0) - (1.0 - s)) / sqrt(s);
    return 0.5 * erfc(z * M_SQRT1_2);
}

void perform_likelihood_ratio_test(double* ohlcv_long, int len_long, int len_short, double dt, RegimeTestStats* out) {
    SVCJParams p_long, p_short;
    optimize_svcj(ohlcv_long, len_long, dt, &p_long, NULL, NULL);
    int offset = len_long - len_short;
    optimize_svcj(ohlcv_long + offset*N_COLS, len_short, dt, &p_short, NULL, NULL);
    
    double* ret_short = malloc((len_short-1)*sizeof(double));
    compute_log_returns(ohlcv_long + offset*N_COLS, len_short, ret_short);
    
    out->ll_constrained = raw_likelihood(ret_short, len_short-1, dt, &p_long);
    out->ll_unconstrained = raw_likelihood(ret_short, len_short-1, dt, &p_short);
    free(ret_short);
    
    out->test_statistic = 2.0 * (out->ll_unconstrained - out->ll_constrained);
    if(out->test_statistic < 0) out->test_statistic = 0;
    out->p_value = chi2_p_value(out->test_statistic, 5);
    out->is_significant = (out->p_value < 0.05) ? 1 : 0;
}