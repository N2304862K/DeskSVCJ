#include "svcj.h"
#include <float.h>

// --- MATH UTILS ---
int cmp(const void* a, const void* b) {
    double x=*(double*)a, y=*(double*)b;
    return (x<y)?-1:(x>y);
}

// Anderson-Darling Test (Tail Sensitive)
// Returns A^2 statistic. Critical value for 95% approx 2.5 in finance context
double perform_anderson_darling(double* data, int n) {
    if (n < 5) return 0.0;
    
    // Normalize data (Standardize)
    double mean=0, sq=0;
    for(int i=0;i<n;i++) mean+=data[i]; mean/=n;
    for(int i=0;i<n;i++) sq+=(data[i]-mean)*(data[i]-mean);
    double std = sqrt(sq/(n-1));
    if(std < 1e-9) return 0.0;
    
    double* sorted = malloc(n*sizeof(double));
    for(int i=0;i<n;i++) sorted[i] = (data[i]-mean)/std;
    qsort(sorted, n, sizeof(double), cmp);
    
    double s = 0;
    for(int i=0; i<n; i++) {
        // CDF of Normal
        double x = sorted[i];
        double cdf = 0.5 * erfc(-x * 0.70710678);
        if(cdf < 1e-9) cdf=1e-9; if(cdf > 0.999999999) cdf=0.999999999;
        
        double term = (2.0*(i+1)-1.0) * (log(cdf) + log(1.0 - sorted[n-1-i])); // Logic check: should be cdf of reversed?
        // Standard AD formula: S = -N - (1/N) * Sum( (2i-1)*(ln(Yi) + ln(1-Y(n+1-i))) )
        // Using correct indexing:
        double cdf_rev = 0.5 * erfc(-sorted[n-1-i] * 0.70710678);
        if(cdf_rev > 0.999999999) cdf_rev=0.999999999;
        
        s += (2.0*(i+1)-1.0) * (log(cdf) + log(1.0 - cdf_rev));
    }
    free(sorted);
    return -n - (1.0/n)*s;
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

// --- CORE PHYSICS ---
void compute_log_returns(double* ohlcv, int n, double* out) {
    for(int i=1; i<n; i++) out[i-1] = log(ohlcv[i*N_COLS+3]/ohlcv[(i-1)*N_COLS+3]);
}

void compute_volume_weighted_returns(double* ohlcv, int n, double* out) {
    double sum_v=0; for(int i=0;i<n;i++) sum_v+=ohlcv[i*N_COLS+4];
    double avg_v = sum_v/n; if(avg_v<1) avg_v=1;
    
    for(int i=1; i<n; i++) {
        double r = log(ohlcv[i*N_COLS+3]/ohlcv[(i-1)*N_COLS+3]);
        double v = ohlcv[i*N_COLS+4];
        // Square Root Law: Variance ~ Volume -> Volatility ~ Sqrt(Volume)
        out[i-1] = r * sqrt(v/avg_v); 
    }
}

// --- OPTIMIZATION & HESSIAN ---
double ukf_likelihood(double* ret, int n, double dt, SVCJParams* p) {
    double ll=0; double v=p->theta; 
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        double v_pred = v + p->kappa*(p->theta - v)*dt;
        if(v_pred<1e-9) v_pred=1e-9;
        double y = ret[t] - (p->mu - 0.5*v_pred)*dt;
        
        // Robust PDF
        double tot_var = v_pred + p->lambda_j*dt*var_j;
        double sd = sqrt(tot_var*dt);
        if(sd<1e-9) sd=1e-9;
        
        double pdf = (1.0/(sqrt(2*M_PI)*sd)) * exp(-0.5*y*y/(sd*sd));
        ll += log(pdf + 1e-25);
        
        // Update (Simplified for Gradient Stability in Hessian)
        v = v_pred + (p->rho*p->sigma_v*dt/tot_var)*y; // Simple Gain
        if(v<1e-9) v=1e-9; if(v>20.0) v=20.0;
    }
    return ll;
}

void check_bounds(SVCJParams* p) {
    if(p->theta<1e-5)p->theta=1e-5; if(p->theta>20)p->theta=20;
    if(p->kappa<0.1)p->kappa=0.1; if(p->kappa>50)p->kappa=50;
    if(p->sigma_v<0.01)p->sigma_v=0.01; if(p->sigma_v>20)p->sigma_v=20;
    if(p->rho>0.99)p->rho=0.99; if(p->rho<-0.99)p->rho=-0.99;
    if(p->lambda_j<0.01)p->lambda_j=0.01;
}

void optimize_svcj(double* ret, int n, double dt, SVCJParams* p) {
    // Quick Estimation
    double sum_sq=0; for(int i=0;i<n;i++) sum_sq+=ret[i]*ret[i];
    p->theta = (sum_sq/n)/dt; 
    p->mu=0; p->kappa=3.0; p->sigma_v=0.5; p->rho=-0.5; 
    p->lambda_j=0.5; p->mu_j=0; p->sigma_j=sqrt(p->theta);
    
    // Condensed Nelder-Mead
    int nd=5; double sim[6][5]; double sc[6];
    for(int i=0;i<=nd;i++) {
        SVCJParams t=*p;
        if(i==1)t.kappa*=1.2; if(i==2)t.theta*=1.2; if(i==3)t.sigma_v*=1.2;
        if(i==4)t.rho+=0.2; if(i==5)t.lambda_j*=1.2;
        check_bounds(&t);
        sim[i][0]=t.kappa; sim[i][1]=t.theta; sim[i][2]=t.sigma_v; 
        sim[i][3]=t.rho; sim[i][4]=t.lambda_j;
        sc[i] = ukf_likelihood(ret, n, dt, &t);
    }
    
    for(int k=0;k<NM_ITER;k++) {
        int vs[6]; for(int j=0;j<6;j++)vs[j]=j;
        for(int i=0;i<6;i++) for(int j=i+1;j<6;j++) if(sc[vs[j]]>sc[vs[i]]) {int tp=vs[i]; vs[i]=vs[j]; vs[j]=tp;}
        double c[5]={0}; for(int i=0;i<5;i++) for(int d=0;d<5;d++) c[d]+=sim[vs[i]][d]; for(int d=0;d<5;d++) c[d]/=5;
        SVCJParams rp=*p; double ref[5];
        for(int d=0;d<5;d++) ref[d]=c[d]+(c[d]-sim[vs[5]][d]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        check_bounds(&rp);
        double rsc=ukf_likelihood(ret, n, dt, &rp);
        if(rsc>sc[vs[0]]) { for(int d=0;d<5;d++) sim[vs[5]][d]=ref[d]; sc[vs[5]]=rsc; }
        else {
            SVCJParams cp=*p; 
            for(int d=0;d<5;d++) sim[vs[5]][d] = c[d] + 0.5*(sim[vs[5]][d]-c[d]);
            cp.kappa=sim[vs[5]][0]; cp.theta=sim[vs[5]][1]; cp.sigma_v=sim[vs[5]][2]; 
            cp.rho=sim[vs[5]][3]; cp.lambda_j=sim[vs[5]][4]; check_bounds(&cp);
            sc[vs[5]] = ukf_likelihood(ret, n, dt, &cp);
        }
    }
    int b=0; for(int i=1;i<6;i++) if(sc[i]>sc[b]) b=i;
    p->kappa=sim[b][0]; p->theta=sim[b][1]; p->sigma_v=sim[b][2]; p->rho=sim[b][3]; p->lambda_j=sim[b][4];
}

// *** HESSIAN & CONFIDENCE ***
void calculate_hessian_errors(double* ret, int n, double dt, SVCJParams* p, HessianMetrics* out) {
    // Compute 2nd Derivative of Likelihood w.r.t parameters
    // We specifically care about Theta, Kappa, Sigma_V
    double eps = 1e-4;
    SVCJParams base = *p;
    double ll_0 = ukf_likelihood(ret, n, dt, &base);
    
    // Diagonal approximation of Hessian (Inverse of Variance)
    // d2L/dTheta2 ~ (L(t+e) - 2L(t) + L(t-e)) / e^2
    
    // Theta
    base.theta = p->theta + eps; double l_p = ukf_likelihood(ret, n, dt, &base);
    base.theta = p->theta - eps; double l_m = ukf_likelihood(ret, n, dt, &base);
    base.theta = p->theta; // Reset
    double d2_theta = (l_p - 2*ll_0 + l_m) / (eps*eps);
    
    // Fisher Information = -E[Hessian]. Since we maximize LL, curvature is negative at peak.
    // Variance = 1 / |Curvature|
    out->se_theta = (fabs(d2_theta) > 1e-6) ? sqrt(1.0 / fabs(d2_theta)) : 100.0;
    
    // Kappa
    base.kappa = p->kappa + eps; l_p = ukf_likelihood(ret, n, dt, &base);
    base.kappa = p->kappa - eps; l_m = ukf_likelihood(ret, n, dt, &base);
    base.kappa = p->kappa;
    double d2_kappa = (l_p - 2*ll_0 + l_m) / (eps*eps);
    out->se_kappa = (fabs(d2_kappa) > 1e-6) ? sqrt(1.0 / fabs(d2_kappa)) : 100.0;
}

// --- ENGINES ---
void run_vov_scan(double* ohlcv, int total_len, double dt, int step, VoVPoint* out_buf, int max_steps) {
    SVCJParams p;
    int idx = 0;
    // Scan [60 ... N]
    for(int w=60; w<total_len; w+=step) {
        if(idx >= max_steps) break;
        // Use Volume-Weighted Returns for fitting
        double* ret = malloc((w-1)*sizeof(double));
        compute_volume_weighted_returns(ohlcv + (total_len-w)*N_COLS, w, ret);
        
        optimize_svcj(ret, w-1, dt, &p);
        
        out_buf[idx].window = w;
        out_buf[idx].sigma_v = p.sigma_v;
        out_buf[idx].theta = p.theta;
        idx++;
        free(ret);
    }
    if(idx < max_steps) out_buf[idx].window = 0;
}

void run_fidelity_pipeline(double* ohlcv, int total_len, double dt, FidelityMetrics* out) {
    // 1. VoV Scan to find Gravity Window (Internal Logic or Passed? Let's do simplified scan here)
    // For efficiency, assume Gravity is pre-determined or use heuristic.
    // Better: Python passes specific windows. BUT prompt asks for coherent flow.
    // Let's implement a quick internal search for Min Sigma_V.
    
    int w_grav = 100; // default
    double min_sig = 1e9;
    
    // Coarse scan: 60, 90, 120, 150, 200, 250
    int candidates[] = {60, 90, 120, 150, 200, 250};
    for(int k=0; k<6; k++) {
        if(candidates[k] > total_len - 30) break;
        int w = candidates[k];
        double* r = malloc(w*sizeof(double));
        compute_volume_weighted_returns(ohlcv + (total_len-w)*N_COLS, w, r);
        SVCJParams p_tmp; optimize_svcj(r, w-1, dt, &p_tmp);
        if(p_tmp.sigma_v < min_sig) { min_sig = p_tmp.sigma_v; w_grav = w; }
        free(r);
    }
    
    // 2. Fixed Impulse
    int w_imp = 30;
    
    // 3. Prepare Data
    // Track A: Gravity (Volume Weighted) for Physics
    double* r_grav_vol = malloc(w_grav*sizeof(double));
    compute_volume_weighted_returns(ohlcv + (total_len-w_grav)*N_COLS, w_grav, r_grav_vol);
    
    // Track B: Impulse (Volume Weighted) for Physics
    double* r_imp_vol = malloc(w_imp*sizeof(double));
    compute_volume_weighted_returns(ohlcv + (total_len-w_imp)*N_COLS, w_imp, r_imp_vol);
    
    // Track C: Impulse (RAW) for Statistical Tests
    double* r_imp_raw = malloc(w_imp*sizeof(double));
    compute_log_returns(ohlcv + (total_len-w_imp)*N_COLS, w_imp, r_imp_raw);
    
    // 4. Fit Physics
    SVCJParams p_grav; optimize_svcj(r_grav_vol, w_grav-1, dt, &p_grav);
    HessianMetrics h_grav; calculate_hessian_errors(r_grav_vol, w_grav-1, dt, &p_grav, &h_grav);
    
    SVCJParams p_imp; optimize_svcj(r_imp_vol, w_imp-1, dt, &p_imp);
    
    // 5. Comparison
    out->win_gravity = w_grav;
    out->win_impulse = w_imp;
    out->theta_gravity = p_grav.theta;
    out->theta_impulse = p_imp.theta;
    out->energy_ratio = p_imp.theta / p_grav.theta;
    
    // Gate 1: Parameter Z-Score
    out->theta_std_err = h_grav.se_theta;
    double diff = fabs(p_imp.theta - p_grav.theta);
    out->param_z_score = diff / (h_grav.se_theta + 1e-9);
    
    // Gate 2: Anderson-Darling (On Raw Returns - are tails fatter?)
    out->ad_stat = perform_anderson_darling(r_imp_raw, w_imp-1);
    
    // Gate 3: Hurst (On Raw Returns - is it trending?)
    out->hurst = calc_hurst(r_imp_raw, w_imp-1);
    
    // Bias
    double bias=0; for(int i=0;i<w_imp-1;i++) bias+=r_imp_raw[i];
    out->residue_bias = bias;
    
    // 6. Decision Logic
    // Valid if:
    // - Param Z > 1.96 (Statistically different structure)
    // - AD > 1.5 (Tails are active)
    // - Hurst > 0.55 (Trend is real)
    
    int c1 = (out->param_z_score > 1.96);
    int c2 = (out->ad_stat > 1.5);
    int c3 = (out->hurst > 0.55);
    
    out->is_valid = (c1 && c2 && c3) ? 1 : 0;
    
    free(r_grav_vol); free(r_imp_vol); free(r_imp_raw);
}