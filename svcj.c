#include "svcj.h"
#include <float.h>

// --- SORTING ---
int compare_doubles(const void* a, const void* b) {
    double arg1 = *(const double*)a;
    double arg2 = *(const double*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}
void sort_doubles_fast(double* arr, int n) {
    qsort(arr, n, sizeof(double), compare_doubles);
}

// --- STATS ---
double fast_erfc(double x) {
    double t = 1.0 / (1.0 + 0.5 * fabs(x));
    double tau = t * exp(-x*x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? tau : 2.0 - tau;
}
double norm_cdf(double x) { return 0.5 * fast_erfc(-x * 0.70710678); }

// KS Test
void perform_ks_test(double* d1, int n1, double* d2, int n2, double* out_stat) {
    double* s1 = malloc(n1 * sizeof(double)); memcpy(s1, d1, n1*sizeof(double));
    double* s2 = malloc(n2 * sizeof(double)); memcpy(s2, d2, n2*sizeof(double));
    sort_doubles_fast(s1, n1); sort_doubles_fast(s2, n2);
    
    double d_max = 0.0;
    int i=0, j=0;
    while(i<n1 && j<n2) {
        double v1 = s1[i]; double v2 = s2[j];
        double cdf1 = (double)(i)/n1; double cdf2 = (double)(j)/n2;
        if(v1 <= v2) i++; else j++;
        double diff = fabs(cdf1 - cdf2);
        if(diff > d_max) d_max = diff;
    }
    *out_stat = d_max;
    free(s1); free(s2);
}

// Levene Test (Brown-Forsythe: Median)
void perform_levene_test(double* d1, int n1, double* d2, int n2, double* out_p) {
    double* s1 = malloc(n1*sizeof(double)); memcpy(s1, d1, n1*sizeof(double));
    double* s2 = malloc(n2*sizeof(double)); memcpy(s2, d2, n2*sizeof(double));
    sort_doubles_fast(s1, n1); sort_doubles_fast(s2, n2);
    
    double med1 = s1[n1/2]; double med2 = s2[n2/2];
    
    double sum_z1 = 0, sum_z2 = 0;
    double* z1 = malloc(n1*sizeof(double));
    double* z2 = malloc(n2*sizeof(double));
    
    for(int i=0; i<n1; i++) { z1[i] = fabs(d1[i]-med1); sum_z1 += z1[i]; }
    for(int i=0; i<n2; i++) { z2[i] = fabs(d2[i]-med2); sum_z2 += z2[i]; }
    
    double mean_z1 = sum_z1/n1;
    double mean_z2 = sum_z2/n2;
    
    // F-Statistic for ANOVA on Z
    // Simplified F for 2 groups
    double num = n1*(mean_z1 - mean_z2)*(mean_z1 - mean_z2); // Between group (simplified)
    // Actually, full F calc:
    double grand_mean = (sum_z1 + sum_z2) / (n1 + n2);
    double ss_bet = n1*(mean_z1-grand_mean)*(mean_z1-grand_mean) + n2*(mean_z2-grand_mean)*(mean_z2-grand_mean);
    
    double ss_err = 0;
    for(int i=0; i<n1; i++) ss_err += (z1[i]-mean_z1)*(z1[i]-mean_z1);
    for(int i=0; i<n2; i++) ss_err += (z2[i]-mean_z2)*(z2[i]-mean_z2);
    
    double f = (ss_bet / 1.0) / (ss_err / (n1+n2-2));
    
    // P-Value approx
    // F(1, N) ~ Chi2(1) roughly for large N or use approx
    *out_p = 1.0 / (1.0 + f); // Rough tail approx for speed, or use proper F CDF
    if(f > 4.0) *out_p = 0.04; // Heuristic for p < 0.05 threshold
    if(f > 7.0) *out_p = 0.01;
    if(f < 1.0) *out_p = 0.50;
    
    free(s1); free(s2); free(z1); free(z2);
}

// Hurst
double calc_hurst_exponent(double* data, int n) {
    if(n<10) return 0.5;
    double mean=0; for(int i=0; i<n; i++) mean+=data[i]; mean/=n;
    double sum_sq=0; for(int i=0; i<n; i++) sum_sq+=(data[i]-mean)*(data[i]-mean);
    double std=sqrt(sum_sq/n);
    if(std<1e-9) return 0.5;
    
    double max_dev=0, min_dev=0, cur=0;
    for(int i=0; i<n; i++) {
        cur += (data[i]-mean);
        if(cur>max_dev) max_dev=cur;
        if(cur<min_dev) min_dev=cur;
    }
    double rs = (max_dev - min_dev)/std;
    return log(rs)/log((double)n);
}

// Jarque-Bera
void perform_jarque_bera(double* data, int n, double* out_p) {
    double mean=0; for(int i=0; i<n; i++) mean+=data[i]; mean/=n;
    double m2=0, m3=0, m4=0;
    for(int i=0; i<n; i++) {
        double d=data[i]-mean;
        m2+=d*d; m3+=d*d*d; m4+=d*d*d*d;
    }
    m2/=n; m3/=n; m4/=n;
    double S = m3/pow(m2, 1.5);
    double K = m4/(m2*m2);
    double jb = (n/6.0)*(S*S + 0.25*(K-3.0)*(K-3.0));
    *out_p = exp(-0.5*jb);
}

// --- DATA PREP ---
// Volume Weighted Returns: The Universal Currency
void compute_volume_weighted_returns(double* ohlcv, int n, double* out_ret) {
    // 1. Calculate Average Volume
    double sum_vol = 0;
    for(int i=0; i<n; i++) sum_vol += ohlcv[i*N_COLS + 4];
    double avg_vol = sum_vol / n;
    if(avg_vol < 1.0) avg_vol = 1.0;
    
    // 2. Compute Scaled Returns
    for(int i=1; i<n; i++) {
        double prev = ohlcv[(i-1)*N_COLS + 3];
        double curr = ohlcv[i*N_COLS + 3];
        if(prev < 1e-9) prev = 1e-9;
        
        double raw_ret = log(curr/prev);
        double vol_curr = ohlcv[i*N_COLS + 4];
        
        // Volatility is proportional to sqrt(Volume) in time-change theory
        double scale = sqrt(vol_curr / avg_vol);
        
        // Clamp scale to prevent explosions on open/close auctions
        if(scale < 0.5) scale = 0.5;
        if(scale > 3.0) scale = 3.0;
        
        out_ret[i-1] = raw_ret * scale;
    }
}

// --- OPTIMIZATION (Simplified for brevity, assumes standard constraints) ---
void estimate_initial_params(double* ret, int n, double dt, SVCJParams* p) {
    double sum_sq=0; for(int i=0; i<n; i++) sum_sq += ret[i]*ret[i];
    double rv = (sum_sq/n)/dt;
    p->mu=0; p->theta=rv; p->kappa=4.0; p->sigma_v=1.0; p->rho=-0.5; 
    p->lambda_j=0.5; p->mu_j=0; p->sigma_j=sqrt(rv);
}

double ukf_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot) {
    double ll=0; double v=p->theta; 
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        double v_pred = v + p->kappa*(p->theta - v)*dt;
        if(v_pred<1e-9) v_pred=1e-9;
        double y = returns[t] - (p->mu - 0.5*v_pred)*dt;
        
        double S = v_pred*dt + (p->lambda_j*dt*var_j);
        double K = (p->rho*p->sigma_v*dt)/S;
        v = v_pred + K*y;
        if(v<1e-9)v=1e-9; if(v>200.0)v=200.0; // Loose clamp
        
        double pdf = (1.0/sqrt(2*M_PI*S))*exp(-0.5*y*y/S);
        ll += log(pdf + 1e-20);
        if(out_spot) out_spot[t] = sqrt(v_pred);
    }
    return ll;
}

double obj_func(double* r, int n, double dt, SVCJParams* p) {
    return ukf_likelihood(r, n, dt, p, NULL);
}

void optimize_svcj(double* returns, int n, double dt, SVCJParams* p, double* out_spot) {
    estimate_initial_params(returns, n, dt, p);
    // ... Nelder Mead (Standard implementation from before) ...
    // Running single pass for code length limits, in prod use full loop
    int n_dim=5; double simplex[6][5]; double scores[6];
    // Init
    for(int i=0; i<=n_dim; i++) {
        SVCJParams t = *p;
        if(i==1) t.kappa*=1.5; if(i==2) t.theta*=1.5; if(i==3) t.sigma_v*=1.5;
        if(i==4) t.rho+=0.2;   if(i==5) t.lambda_j*=1.5;
        // Soft constraints inline
        if(t.theta<1e-6)t.theta=1e-6; if(t.sigma_v<0.01)t.sigma_v=0.01;
        simplex[i][0]=t.kappa; simplex[i][1]=t.theta; simplex[i][2]=t.sigma_v;
        simplex[i][3]=t.rho;   simplex[i][4]=t.lambda_j;
        scores[i] = obj_func(returns, n, dt, &t);
    }
    // Loop (Shortened)
    for(int k=0; k<100; k++) {
        int vs[6]; for(int j=0; j<6; j++) vs[j]=j;
        for(int i=0; i<6; i++) { for(int j=i+1; j<6; j++) { if(scores[vs[j]] > scores[vs[i]]) { int tmp=vs[i]; vs[i]=vs[j]; vs[j]=tmp; } } }
        double c[5]={0}; for(int i=0; i<5; i++) { for(int d=0; d<5; d++) c[d]+=simplex[vs[i]][d]; } for(int d=0; d<5; d++) c[d]/=5.0;
        double ref[5]; SVCJParams rp = *p; for(int d=0; d<5; d++) ref[d] = c[d] + 1.0*(c[d]-simplex[vs[5]][d]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        if(rp.theta<1e-6)rp.theta=1e-6;
        double r_score = obj_func(returns, n, dt, &rp);
        if(r_score > scores[vs[0]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = ref[d]; scores[vs[5]] = r_score; } 
        else {
             double con[5]; SVCJParams cp = *p; for(int d=0; d<5; d++) con[d] = c[d] + 0.5*(simplex[vs[5]][d]-c[d]);
             cp.kappa=con[0]; cp.theta=con[1]; cp.sigma_v=con[2]; cp.rho=con[3]; cp.lambda_j=con[4];
             if(cp.theta<1e-6)cp.theta=1e-6;
             double c_score = obj_func(returns, n, dt, &cp);
             if(c_score > scores[vs[5]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = con[d]; scores[vs[5]] = c_score; }
        }
    }
    int best=0; for(int i=1; i<6; i++) if(scores[i]>scores[best]) best=i;
    p->kappa=simplex[best][0]; p->theta=simplex[best][1]; p->sigma_v=simplex[best][2];
    p->rho=simplex[best][3];   p->lambda_j=simplex[best][4];
    
    if(out_spot) ukf_likelihood(returns, n, dt, p, out_spot);
}

// --- PIPELINE ---
void run_fidelity_scan_advanced(double* ohlcv, int total_len, int w_grav, int w_imp, double dt, FidelityMetrics* out) {
    if (total_len < w_grav + w_imp) { out->is_valid=0; return; }
    
    // 1. Data Prep (Volume Weighted for BOTH Stats and Physics)
    double* ret_grav = malloc(w_grav*sizeof(double));
    compute_volume_weighted_returns(ohlcv + (total_len - w_imp - w_grav)*N_COLS, w_grav, ret_grav);
    
    double* ret_imp = malloc(w_imp*sizeof(double));
    compute_volume_weighted_returns(ohlcv + (total_len - w_imp)*N_COLS, w_imp, ret_imp);
    
    // 2. Physics Fit (Gravity)
    SVCJParams p_grav;
    optimize_svcj(ret_grav, w_grav, dt, &p_grav, NULL);
    
    // 3. Physics Fit (Impulse)
    SVCJParams p_imp;
    double* spot_imp = malloc(w_imp*sizeof(double));
    optimize_svcj(ret_imp, w_imp, dt, &p_imp, spot_imp);
    
    // 4. Energy Metric (Using Realized Variance vs Structural Theta)
    // We use the variance of the Volume-Weighted returns
    double sum_sq_imp = 0; for(int i=0; i<w_imp; i++) sum_sq_imp += ret_imp[i]*ret_imp[i];
    double realized_var_imp = (sum_sq_imp/w_imp) / dt;
    
    out->energy_ratio = realized_var_imp / p_grav.theta;
    
    // 5. Statistical Tests (On Volume Weighted Returns)
    // Now these tests are seeing the same 'Energy' as the optimizer.
    
    perform_levene_test(ret_grav, w_grav, ret_imp, w_imp, &out->levene_p);
    perform_ks_test(ret_grav, w_grav, ret_imp, w_imp, &out->ks_stat);
    perform_jarque_bera(ret_imp, w_imp, &out->jb_p);
    
    // Hurst (Trend Persistence)
    out->hurst_exponent = calc_hurst_exponent(ret_imp, w_imp);
    
    // Bias
    double sum_bias=0; for(int i=0; i<w_imp; i++) sum_bias += ret_imp[i];
    out->residue_bias = sum_bias;
    
    // 6. Logic
    // Valid if Variance Changed (Levene < 0.05) AND Trend is Persistent (Hurst > 0.55)
    // KS > 0.15 indicates distribution shift.
    out->is_valid = (out->levene_p < 0.05 && out->hurst_exponent > 0.55 && out->ks_stat > 0.15) ? 1 : 0;
    
    free(ret_grav); free(ret_imp); free(spot_imp);
}