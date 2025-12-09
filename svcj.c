#include "svcj.h"
#include <float.h>

// --- MATH HELPERS ---
double fast_erfc(double x) {
    double t = 1.0 / (1.0 + 0.5 * fabs(x));
    double tau = t * exp(-x*x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? tau : 2.0 - tau;
}
double norm_cdf(double x) { return 0.5 * fast_erfc(-x * 0.70710678); }

// Standard F-Test CDF (Approximation)
double f_test_p_value(double F, int df1, int df2) {
    if (F <= 0) return 1.0;
    // Paulson's approximation
    double a = 2.0 / (9.0 * df1);
    double b = 2.0 / (9.0 * df2);
    double y = pow(F, 1.0/3.0);
    double z = ((1.0-b)*y - (1.0-a)) / sqrt(b*y*y + a);
    return 2.0 * norm_cdf(-z); // One-tailed (We care if Var1 > Var2)
}

int compare_doubles(const void* a, const void* b) {
    double arg1 = *(const double*)a; double arg2 = *(const double*)b;
    return (arg1 > arg2) - (arg1 < arg2);
}

// --- CORE PHYSICS ---

void compute_volume_weighted_returns(double* ohlcv, int n, double* out_ret) {
    // 1. Calc Avg Volume (Safety Clean)
    double sum_vol = 0;
    int valid_count = 0;
    for(int i=0; i<n; i++) {
        double v = ohlcv[i*N_COLS + 4];
        if (v > 0) { sum_vol += v; valid_count++; }
    }
    double avg_vol = (valid_count > 0) ? sum_vol / valid_count : 1.0;
    
    // 2. Weight Returns
    for(int i=1; i<n; i++) {
        double prev = ohlcv[(i-1)*N_COLS + 3];
        double curr = ohlcv[i*N_COLS + 3];
        double vol = ohlcv[i*N_COLS + 4];
        
        if (prev < 1e-9) prev = 1e-9;
        double raw = log(curr / prev);
        
        // Scale: sqrt(RelativeVolume)
        // High Volume = Higher Effective Variance
        double scale = sqrt(vol / avg_vol);
        
        // Safety Clamps to prevent NaN
        if (isnan(scale) || isinf(scale)) scale = 1.0;
        if (scale < 0.1) scale = 0.1; 
        if (scale > 10.0) scale = 10.0;
        
        out_ret[i-1] = raw * scale;
    }
}

void check_constraints(SVCJParams* p) {
    if(p->kappa < 0.01) p->kappa = 0.01; if(p->kappa > 100.0) p->kappa = 100.0;
    if(p->theta < 1e-5) p->theta = 1e-5; if(p->theta > 50.0) p->theta = 50.0; // Hard Floor
    if(p->sigma_v < 0.01) p->sigma_v = 0.01; if(p->sigma_v > 50.0) p->sigma_v = 50.0;
    if(p->rho > 0.99) p->rho = 0.99; if(p->rho < -0.99) p->rho = -0.99;
    if(p->lambda_j < 0.01) p->lambda_j = 0.01; if(p->lambda_j > 2000.0) p->lambda_j = 2000.0;
    if(p->sigma_j < 0.001) p->sigma_j = 0.001;
    
    double feller = 2.0 * p->kappa * p->theta;
    if (p->sigma_v*p->sigma_v > feller*20.0) p->sigma_v = sqrt(feller*20.0);
}

void estimate_initial_params(double* ret, int n, double dt, SVCJParams* p) {
    double sum_sq = 0;
    for(int i=0; i<n; i++) sum_sq += ret[i]*ret[i];
    double rv = (sum_sq / n) / dt;
    if (rv < 1e-5) rv = 1e-5;
    
    p->mu=0; p->theta=rv; p->kappa=4.0; p->sigma_v=sqrt(rv); 
    p->rho=-0.5; p->lambda_j=0.5; p->mu_j=0; p->sigma_j=sqrt(rv);
    check_constraints(p);
}

double ukf_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot) {
    double ll = 0.0;
    double v = p->theta;
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        double v_pred = v + p->kappa*(p->theta - v)*dt;
        if(v_pred < 1e-9) v_pred = 1e-9;
        
        double drift = p->mu - 0.5*v_pred;
        double y = returns[t] - drift*dt;
        
        double S = v_pred*dt + (p->lambda_j*dt*var_j);
        double K = (p->rho*p->sigma_v*dt)/S;
        
        v = v_pred + K*y;
        if(v < 1e-9) v = 1e-9; if(v > 50.0) v = 50.0;
        
        // Likelihood
        double pdf = (1.0/sqrt(2*M_PI*S)) * exp(-0.5*y*y/S);
        ll += log(pdf + 1e-20);
        
        if(out_spot) out_spot[t] = sqrt(v_pred);
    }
    return ll;
}

double obj_func(double* r, int n, double dt, SVCJParams* p) {
    // Penalty for bad params
    return ukf_likelihood(r, n, dt, p, NULL) - 0.1*p->sigma_v*p->sigma_v;
}

void optimize_svcj_volume(double* returns, double* volumes, int n, double dt, SVCJParams* p, double* out_spot) {
    // Note: Volumes used in pre-calc returns now, simpler interface
    estimate_initial_params(returns, n, dt, p);
    
    int n_dim=5; double simplex[6][5]; double scores[6];
    for(int i=0; i<=n_dim; i++) {
        SVCJParams t = *p;
        if(i==1) t.kappa*=1.5; if(i==2) t.theta*=1.5; if(i==3) t.sigma_v*=1.5;
        if(i==4) t.rho+=0.2;   if(i==5) t.lambda_j*=1.5;
        check_constraints(&t);
        simplex[i][0]=t.kappa; simplex[i][1]=t.theta; simplex[i][2]=t.sigma_v;
        simplex[i][3]=t.rho;   simplex[i][4]=t.lambda_j;
        scores[i] = obj_func(returns, n, dt, &t);
    }
    
    for(int k=0; k<150; k++) {
        int vs[6]; for(int j=0; j<6; j++) vs[j]=j;
        for(int i=0; i<6; i++) { for(int j=i+1; j<6; j++) { if(scores[vs[j]] > scores[vs[i]]) { int tmp=vs[i]; vs[i]=vs[j]; vs[j]=tmp; } } }
        double c[5]={0}; for(int i=0; i<5; i++) { for(int d=0; d<5; d++) c[d]+=simplex[vs[i]][d]; } for(int d=0; d<5; d++) c[d]/=5.0;
        double ref[5]; SVCJParams rp = *p; for(int d=0; d<5; d++) ref[d] = c[d] + 1.0*(c[d]-simplex[vs[5]][d]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        check_constraints(&rp);
        double r_score = obj_func(returns, n, dt, &rp);
        if(r_score > scores[vs[0]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = ref[d]; scores[vs[5]] = r_score; } 
        else {
             double con[5]; SVCJParams cp = *p; for(int d=0; d<5; d++) con[d] = c[d] + 0.5*(simplex[vs[5]][d]-c[d]);
             cp.kappa=con[0]; cp.theta=con[1]; cp.sigma_v=con[2]; cp.rho=con[3]; cp.lambda_j=con[4];
             check_constraints(&cp);
             double c_score = obj_func(returns, n, dt, &cp);
             if(c_score > scores[vs[5]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = con[d]; scores[vs[5]] = c_score; }
        }
    }
    int best=0; for(int i=1; i<6; i++) if(scores[i]>scores[best]) best=i;
    p->kappa=simplex[best][0]; p->theta=simplex[best][1]; p->sigma_v=simplex[best][2];
    p->rho=simplex[best][3];   p->lambda_j=simplex[best][4];
    
    if(out_spot) ukf_likelihood(returns, n, dt, p, out_spot);
}

// --- STATISTICAL TESTS ---
void perform_ks_test(double* d1, int n1, double* d2, int n2, double* out_stat) {
    double* s1 = malloc(n1*sizeof(double)); memcpy(s1, d1, n1*sizeof(double));
    double* s2 = malloc(n2*sizeof(double)); memcpy(s2, d2, n2*sizeof(double));
    qsort(s1, n1, sizeof(double), compare_doubles);
    qsort(s2, n2, sizeof(double), compare_doubles);
    
    double d_max = 0.0;
    int i=0, j=0;
    while(i<n1 && j<n2) {
        double v1=s1[i]; double v2=s2[j];
        double cdf1=(double)i/n1; double cdf2=(double)j/n2;
        if(v1<=v2) i++; else j++;
        double diff = fabs(cdf1-cdf2);
        if(diff > d_max) d_max=diff;
    }
    *out_stat = d_max;
    free(s1); free(s2);
}

double calc_hurst(double* data, int n) {
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
    return log((max_dev-min_dev)/std) / log((double)n);
}

void perform_jarque_bera(double* data, int n, double* out_p) {
    double mean=0; for(int i=0; i<n; i++) mean+=data[i]; mean/=n;
    double m2=0, m3=0, m4=0;
    for(int i=0; i<n; i++) {
        double d=data[i]-mean;
        m2+=d*d; m3+=d*d*d; m4+=d*d*d*d;
    }
    m2/=n; m3/=n; m4/=n;
    double S=m3/pow(m2, 1.5); double K=m4/(m2*m2);
    double jb=(n/6.0)*(S*S + 0.25*(K-3.0)*(K-3.0));
    *out_p = exp(-0.5*jb);
}

// --- MAIN SCAN ---
void run_fidelity_scan_advanced(double* ohlcv, int total_len, int w_grav, int w_imp, double dt, FidelityMetrics* out) {
    if (total_len < w_grav + w_imp) { out->is_valid=0; return; }
    
    // 1. Prepare Weighted Data
    double* ret_grav = malloc(w_grav*sizeof(double));
    compute_volume_weighted_returns(ohlcv + (total_len - w_imp - w_grav)*N_COLS, w_grav, ret_grav);
    
    double* ret_imp = malloc(w_imp*sizeof(double));
    compute_volume_weighted_returns(ohlcv + (total_len - w_imp)*N_COLS, w_imp, ret_imp);
    
    // 2. Variance Ratio (Realized Energy)
    // We compare the Raw Weighted Variances directly (Model-Free check)
    double sum_sq_g = 0; for(int i=0; i<w_grav; i++) sum_sq_g += ret_grav[i]*ret_grav[i];
    double var_g = sum_sq_g / (w_grav - 1);
    
    double sum_sq_i = 0; for(int i=0; i<w_imp; i++) sum_sq_i += ret_imp[i]*ret_imp[i];
    double var_i = sum_sq_i / (w_imp - 1);
    
    // Energy Ratio (Realized)
    if (var_g < 1e-9) var_g = 1e-9;
    out->realized_f_stat = var_i / var_g;
    
    // 3. F-Test (Variance Significance)
    out->f_p_value = f_test_p_value(out->realized_f_stat, w_imp-1, w_grav-1);
    
    // 4. Model Fit (Physics)
    // We still run the model to get drift and theta for context
    SVCJParams p_grav;
    optimize_svcj_volume(ret_grav, NULL, w_grav, dt, &p_grav, NULL);
    out->energy_ratio = out->realized_f_stat; // Align concepts
    
    // 5. Direction (Bias)
    double sum_bias=0; for(int i=0; i<w_imp; i++) sum_bias += ret_imp[i];
    out->residue_bias = sum_bias;
    
    // 6. Other Stats
    perform_ks_test(ret_grav, w_grav, ret_imp, w_imp, &out->ks_stat);
    perform_jarque_bera(ret_imp, w_imp, &out->jb_p);
    out->hurst_exponent = calc_hurst(ret_imp, w_imp);
    
    // 7. Logic
    // Breakout = Significant Variance Expansion (p<0.05) + Persistence (H>0.55)
    out->is_valid = (out->f_p_value < 0.05 && out->hurst_exponent > 0.55 && out->ks_stat > 0.15) ? 1 : 0;
    
    free(ret_grav); free(ret_imp);
}