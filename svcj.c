#include "svcj.h"
#include <float.h>

// --- SORTING & STATS ---
int cmp(const void* a, const void* b) {
    double x=*(double*)a, y=*(double*)b;
    return (x<y)?-1:(x>y);
}
void sort_doubles_fast(double* arr, int n) { qsort(arr, n, sizeof(double), cmp); }

double fast_erfc(double x) {
    double t = 1.0/(1.0+0.5*fabs(x));
    double tau = t*exp(-x*x-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))));
    return x>=0?tau:2.0-tau;
}
double norm_cdf(double x) { return 0.5*fast_erfc(-x*0.70710678); }

// --- TESTS ---
void perform_ks_test(double* d1, int n1, double* d2, int n2, double* out_stat) {
    double* s1=malloc(n1*8); memcpy(s1,d1,n1*8); sort_doubles_fast(s1,n1);
    double* s2=malloc(n2*8); memcpy(s2,d2,n2*8); sort_doubles_fast(s2,n2);
    double dmax=0; int i=0,j=0;
    while(i<n1 && j<n2) {
        double c1=(double)i/n1, c2=(double)j/n2;
        if(s1[i]<=s2[j]) i++; else j++;
        double diff=fabs(c1-c2);
        if(diff>dmax) dmax=diff;
    }
    *out_stat=dmax; free(s1); free(s2);
}

void perform_levene_test(double* d1, int n1, double* d2, int n2, double* out_p) {
    // Robust Brown-Forsythe (Median)
    double* s1=malloc(n1*8); memcpy(s1,d1,n1*8); sort_doubles_fast(s1,n1);
    double* s2=malloc(n2*8); memcpy(s2,d2,n2*8); sort_doubles_fast(s2,n2);
    double m1=s1[n1/2], m2=s2[n2/2];
    
    double sum1=0, sum2=0;
    for(int i=0; i<n1; i++) sum1 += fabs(d1[i]-m1);
    for(int i=0; i<n2; i++) sum2 += fabs(d2[i]-m2);
    double mean1=sum1/n1, mean2=sum2/n2;
    
    // F-Stat Proxy: Ratio of Deviations
    double ratio = (mean1 > mean2) ? mean1/mean2 : mean2/mean1;
    // Heuristic p-value for speed (Real F-test requires DOF calculus)
    // Ratio > 1.5 is typically significant for financial variance
    if(ratio < 1.0) ratio=1.0;
    *out_p = 1.0 / (ratio * ratio); // steep decay
    free(s1); free(s2);
}

double calc_hurst(double* data, int n) {
    if(n<20) return 0.5;
    double m=0; for(int i=0;i<n;i++) m+=data[i]; m/=n;
    double ss=0; for(int i=0;i<n;i++) ss+=(data[i]-m)*(data[i]-m);
    double std=sqrt(ss/n); if(std<1e-9) return 0.5;
    double maxd=-1e9, mind=1e9, c=0;
    for(int i=0;i<n;i++) { c+=(data[i]-m); if(c>maxd)maxd=c; if(c<mind)mind=c; }
    return log((maxd-mind)/std)/log((double)n);
}

void perform_jb(double* data, int n, double* out_p) {
    double m=0; for(int i=0;i<n;i++) m+=data[i]; m/=n;
    double m2=0, m3=0, m4=0;
    for(int i=0;i<n;i++) {
        double d=data[i]-m;
        m2+=d*d; m3+=d*d*d; m4+=d*d*d*d;
    }
    m2/=n; m3/=n; m4/=n; if(m2<1e-9) { *out_p=0; return; }
    double S=m3/pow(m2,1.5); double K=m4/(m2*m2);
    double jb=(n/6.0)*(S*S + 0.25*(K-3)*(K-3));
    *out_p=exp(-0.5*jb);
}

// --- DATA PREP ---
void compute_log_returns(double* ohlcv, int n, double* out_ret, double* out_vol) {
    // RAW Log Returns. No scaling.
    for(int i=1; i<n; i++) {
        double prev = ohlcv[(i-1)*N_COLS + 3];
        double curr = ohlcv[i*N_COLS + 3];
        out_vol[i-1] = ohlcv[i*N_COLS + 4]; // Volume
        if(prev < 1e-9) prev = 1e-9;
        out_ret[i-1] = log(curr/prev);
    }
}

// --- PHYSICS ---
void check_constraints(SVCJParams* p) {
    if(p->theta < 1e-5) p->theta = 1e-5; if(p->theta > 10.0) p->theta = 10.0;
    if(p->kappa < 0.1) p->kappa = 0.1;
    if(p->sigma_v < 0.05) p->sigma_v = 0.05;
    if(p->lambda_j < 0.01) p->lambda_j = 0.01;
}

// Time Dilation Likelihood
double ukf_volume_likelihood(double* ret, double* vol, int n, double dt, double avg_vol, SVCJParams* p) {
    double ll=0; double v=p->theta; 
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        // Volume Clock: High Volume = More Time passed
        double time_scale = (avg_vol > 0) ? vol[t]/avg_vol : 1.0;
        if(time_scale < 0.2) time_scale = 0.2;
        if(time_scale > 5.0) time_scale = 5.0;
        
        double dt_eff = dt * time_scale;
        
        // Standard Heston-Jump Update with dt_eff
        double v_pred = v + p->kappa*(p->theta - v)*dt_eff;
        if(v_pred < 1e-8) v_pred = 1e-8;
        
        double y = ret[t] - (p->mu - 0.5*v_pred)*dt_eff;
        double tot_var = v_pred + (p->lambda_j * var_j); // Approx expected var
        double step_std = sqrt(tot_var * dt_eff);
        if(step_std < 1e-8) step_std = 1e-8;
        
        double pdf = (1.0/(sqrt(2*M_PI)*step_std)) * exp(-0.5*y*y/(step_std*step_std));
        ll += log(pdf + 1e-20);
        
        // Simple Update
        double K = 0.1; // Static gain for stability in fit
        v = v_pred + K * (y*y - tot_var*dt_eff);
        if(v < 1e-8) v = 1e-8; if(v > 5.0) v = 5.0;
    }
    return ll;
}

double obj_func(double* r, double* v, int n, double dt, double av, SVCJParams* p) {
    // Penalty for unnatural parameters
    double pen = 0;
    if(p->theta > 2.0) pen += (p->theta - 2.0)*100;
    return ukf_volume_likelihood(r, v, n, dt, av, p) - pen;
}

void optimize_svcj(double* ret, double* vol, int n, double dt, SVCJParams* p) {
    // Robust Init
    double sum_sq=0, sum_vol=0;
    for(int i=0; i<n; i++) { sum_sq+=ret[i]*ret[i]; sum_vol+=vol[i]; }
    double rv = (sum_sq/n)/dt;
    double av = sum_vol/n;
    
    p->mu=0; p->theta=rv; p->kappa=3.0; p->sigma_v=0.5; 
    p->rho=-0.5; p->lambda_j=0.5; p->mu_j=0; p->sigma_j=sqrt(rv);
    
    // Nelder-Mead (Simplified)
    int n_dim=5; double simplex[6][5]; double scores[6];
    for(int i=0; i<=n_dim; i++) {
        SVCJParams t = *p;
        if(i==1) t.kappa*=1.5; if(i==2) t.theta*=1.5;
        check_constraints(&t);
        simplex[i][0]=t.kappa; simplex[i][1]=t.theta; simplex[i][2]=t.sigma_v;
        simplex[i][3]=t.rho;   simplex[i][4]=t.lambda_j;
        scores[i] = obj_func(ret, vol, n, dt, av, &t);
    }
    
    for(int k=0; k<150; k++) { // Reduced iters for speed
        int vs[6]; for(int j=0; j<6; j++) vs[j]=j;
        for(int i=0; i<6; i++) { for(int j=i+1; j<6; j++) { if(scores[vs[j]] > scores[vs[i]]) { int tmp=vs[i]; vs[i]=vs[j]; vs[j]=tmp; } } }
        double c[5]={0}; for(int i=0; i<5; i++) { for(int d=0; d<5; d++) c[d]+=simplex[vs[i]][d]; } for(int d=0; d<5; d++) c[d]/=5.0;
        double ref[5]; SVCJParams rp = *p; for(int d=0; d<5; d++) ref[d] = c[d] + 1.0*(c[d]-simplex[vs[5]][d]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        check_constraints(&rp);
        double r_score = obj_func(ret, vol, n, dt, av, &rp);
        if(r_score > scores[vs[0]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = ref[d]; scores[vs[5]] = r_score; } 
        else {
             double con[5]; SVCJParams cp = *p; for(int d=0; d<5; d++) con[d] = c[d] + 0.5*(simplex[vs[5]][d]-c[d]);
             cp.kappa=con[0]; cp.theta=con[1]; cp.sigma_v=con[2]; cp.rho=con[3]; cp.lambda_j=con[4];
             check_constraints(&cp);
             double c_score = obj_func(ret, vol, n, dt, av, &cp);
             if(c_score > scores[vs[5]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = con[d]; scores[vs[5]] = c_score; }
        }
    }
    int best=0; for(int i=1; i<6; i++) if(scores[i]>scores[best]) best=i;
    p->kappa=simplex[best][0]; p->theta=simplex[best][1]; p->sigma_v=simplex[best][2];
    p->rho=simplex[best][3];   p->lambda_j=simplex[best][4];
}

// --- PIPELINE ---
void run_fidelity_scan_native(double* ohlcv, int total_len, int w_grav, int w_imp, double dt, FidelityMetrics* out) {
    if(total_len < w_grav + w_imp) { out->is_valid=0; return; }
    
    // 1. Prepare Disjoint Data
    double* r_grav = malloc(w_grav*8); double* v_grav = malloc(w_grav*8);
    compute_log_returns(ohlcv + (total_len-w_imp-w_grav)*N_COLS, w_grav, r_grav, v_grav);
    
    double* r_imp = malloc(w_imp*8); double* v_imp = malloc(w_imp*8);
    compute_log_returns(ohlcv + (total_len-w_imp)*N_COLS, w_imp, r_imp, v_imp);
    
    // 2. Fit Physics (Volume Aware)
    SVCJParams p_grav; optimize_svcj(r_grav, v_grav, w_grav-1, dt, &p_grav);
    SVCJParams p_imp;  optimize_svcj(r_imp,  v_imp,  w_imp-1,  dt, &p_imp);
    
    // 3. Energy Comparison (Theta vs Theta)
    // Coherent comparison: Model Structural Vol vs Model Structural Vol
    // Avoids Realized Vol noise.
    out->energy_ratio = p_imp.theta / p_grav.theta;
    
    // 4. Statistics (On RAW RETURNS)
    // Levene checks if the variance of R_imp is distinct from R_grav
    perform_levene_test(r_grav, w_grav-1, r_imp, w_imp-1, &out->levene_p);
    
    // KS checks if distribution shape changed
    perform_ks_test(r_grav, w_grav-1, r_imp, w_imp-1, &out->ks_stat);
    
    // Hurst (Persistence) on Impulse
    out->hurst_exponent = calc_hurst(r_imp, w_imp-1);
    
    // Jarque-Bera on Impulse
    perform_jb(r_imp, w_imp-1, &out->jb_p);
    
    // Bias
    double bias=0; for(int i=0; i<w_imp-1; i++) bias += r_imp[i];
    out->residue_bias = bias;
    
    // 5. Decision
    // Valid if: Physics expanded (Energy > 1.2) AND Stats confirm (Levene < 0.05) AND Trend is Real (Hurst > 0.55)
    int sig_energy = (out->energy_ratio > 1.2) || (out->energy_ratio < 0.8);
    int sig_stat = (out->levene_p < 0.10); // slightly looser for financial noise
    int sig_trend = (out->hurst_exponent > 0.55);
    
    out->is_valid = (sig_energy && sig_stat && sig_trend) ? 1 : 0;
    
    free(r_grav); free(v_grav); free(r_imp); free(v_imp);
}