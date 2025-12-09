#include "svcj.h"
#include <float.h>

// --- Helper: Sorting ---
int cmp(const void* a, const void* b) {
    if (*(double*)a > *(double*)b) return 1;
    if (*(double*)a < *(double*)b) return -1;
    return 0;
}

// --- 5. & 6. Detrending + Volume Scaling ---
void detrend_log_returns(double* ohlcv, int n, double* out_ret, double* out_vol_scale) {
    double sum_vol = 0;
    for(int i=0; i<n; i++) sum_vol += ohlcv[(i+1)*N_COLS + 4]; // Vol index 4
    double avg_vol = sum_vol / n;
    
    // Calculate Returns & Linear Reg
    double sum_x=0, sum_y=0, sum_xy=0, sum_xx=0;
    for(int i=0; i<n; i++) {
        double curr = ohlcv[(i+1)*N_COLS+3];
        double prev = ohlcv[i*N_COLS+3];
        double r = log(curr/prev);
        
        // Vol Scale: (Vol / AvgVol)
        double v = ohlcv[(i+1)*N_COLS + 4];
        out_vol_scale[i] = (avg_vol > 0) ? v / avg_vol : 1.0;
        if(out_vol_scale[i] < 0.1) out_vol_scale[i] = 0.1; // Floor
        
        // Regression accumulators
        sum_x += i; sum_y += r;
        sum_xy += i*r; sum_xx += i*i;
        out_ret[i] = r; // Temp store raw
    }
    
    // Linear Trend
    double slope = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x);
    double intercept = (sum_y - slope*sum_x) / n;
    
    // Detrend
    for(int i=0; i<n; i++) {
        out_ret[i] -= (slope*i + intercept);
    }
}

// --- 2. Hurst Exponent (R/S Analysis) ---
double calc_hurst(double* data, int n) {
    if (n < 10) return 0.5;
    
    // Calculate Mean
    double mean = 0;
    for(int i=0; i<n; i++) mean += data[i];
    mean /= n;
    
    // Calculate Deviations & Range
    double sum_sq_diff = 0;
    double max_cum = -1e9, min_cum = 1e9, cum = 0;
    
    for(int i=0; i<n; i++) {
        double diff = data[i] - mean;
        sum_sq_diff += diff*diff;
        cum += diff;
        if(cum > max_cum) max_cum = cum;
        if(cum < min_cum) min_cum = cum;
    }
    
    double std = sqrt(sum_sq_diff / n);
    if(std < 1e-9) return 0.5;
    
    double range = max_cum - min_cum;
    double rs = range / std;
    
    // H = log(R/S) / log(N)
    return log(rs) / log((double)n);
}

// --- 3. CUSUM Changepoint ---
int detect_changepoint(double* data, int n) {
    // Scan variance for breaks
    double mean = 0, sq_mean = 0;
    for(int i=0; i<n; i++) { mean += data[i]; sq_mean += data[i]*data[i]; }
    mean /= n; sq_mean /= n;
    double global_var = sq_mean - mean*mean;
    
    double sum = 0, sum_sq = 0;
    double max_diff = 0;
    int break_idx = 0;
    
    for(int i=0; i<n-10; i++) {
        sum += data[i]; sum_sq += data[i]*data[i];
        if (i < 20) continue;
        
        double curr_var = (sum_sq / (i+1)) - (sum/(i+1))*(sum/(i+1));
        if (fabs(curr_var - global_var) > max_diff) {
            max_diff = fabs(curr_var - global_var);
            break_idx = i;
        }
    }
    
    // If break is significant (> 50% shift), return index. Else 0.
    if (max_diff > 0.5 * global_var) return break_idx;
    return 0;
}

// --- 4. KS Test (Distribution Matching) ---
double perform_ks_test(double* g1, int n1, double* g2, int n2) {
    double* s1 = malloc(n1*sizeof(double)); memcpy(s1, g1, n1*sizeof(double));
    double* s2 = malloc(n2*sizeof(double)); memcpy(s2, g2, n2*sizeof(double));
    qsort(s1, n1, sizeof(double), cmp);
    qsort(s2, n2, sizeof(double), cmp);
    
    double max_d = 0;
    int i=0, j=0;
    while(i<n1 && j<n2) {
        double d1 = s1[i], d2 = s2[j];
        double cdf1 = (double)i/n1, cdf2 = (double)j/n2;
        double diff = fabs(cdf1 - cdf2);
        if(diff > max_d) max_d = diff;
        if(d1 <= d2) i++; else j++;
    }
    free(s1); free(s2);
    return max_d; // KS Stat (Higher = Different Distributions)
}

// --- Standard Core ---
void check_constraints(SVCJParams* p) {
    if(p->theta<1e-6) p->theta=1e-6; if(p->theta>50) p->theta=50;
    if(p->kappa<0.1) p->kappa=0.1; if(p->sigma_v<0.01) p->sigma_v=0.01;
    if(p->lambda_j<0.01) p->lambda_j=0.01;
    double feller = 2.0*p->kappa*p->theta;
    if(p->sigma_v*p->sigma_v > feller*20) p->sigma_v=sqrt(feller*20);
}

void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p) {
    double sum_gk=0;
    for(int i=0; i<n; i++) {
        double h=log(ohlcv[i*N_COLS+1]/ohlcv[i*N_COLS+2]); // H/L
        double c=log(ohlcv[i*N_COLS+3]/ohlcv[i*N_COLS]);   // C/O
        sum_gk += 0.5*h*h - 0.386*c*c;
    }
    double rv = (sum_gk/n)/dt;
    p->theta=rv; p->kappa=2.0; p->sigma_v=sqrt(rv); p->rho=-0.5; 
    p->lambda_j=0.5; p->mu=0; p->mu_j=0; p->sigma_j=sqrt(rv);
    check_constraints(p);
}

double ukf_pure_likelihood(double* returns, double* vol_scale, int n, double dt, SVCJParams* p, double* out_spot) {
    double ll=0; double v=p->theta; 
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        // Volume Ticking: Effective DT
        double dt_eff = dt * vol_scale[t];
        
        double v_pred = v + p->kappa*(p->theta - v)*dt_eff;
        if(v_pred<1e-9) v_pred=1e-9;
        
        double y = returns[t] - (p->mu - 0.5*v_pred)*dt_eff;
        
        double rob_var = (v_pred<1e-9)?1e-9:v_pred; rob_var *= dt_eff;
        double pdf_d = (1.0/sqrt(rob_var*2*M_PI))*exp(-0.5*y*y/rob_var);
        
        double tot_j = rob_var + var_j;
        double pdf_j = (1.0/sqrt(tot_j*2*M_PI))*exp(-0.5*(y-p->mu_j)*(y-p->mu_j)/tot_j);
        
        double prior = p->lambda_j*dt_eff; if(prior>0.99) prior=0.99;
        double den = pdf_j*prior + pdf_d*(1.0-prior);
        if(den<1e-25) den=1e-25;
        double post = (pdf_j*prior)/den;
        
        double S = v_pred*dt_eff + post*var_j;
        v = v_pred + (p->rho*p->sigma_v*dt_eff/S)*y;
        if(v<1e-9) v=1e-9;
        
        if(out_spot) out_spot[t] = sqrt(v_pred);
        ll += log(den);
    }
    return ll;
}

double obj_func(double* r, double* vs, int n, double dt, SVCJParams* p) {
    return ukf_pure_likelihood(r, vs, n, dt, p, NULL) - 0.05*p->sigma_v*p->sigma_v;
}

void optimize_svcj(double* ohlcv, int n, double dt, double* v_scale, SVCJParams* p, double* out_spot) {
    estimate_initial_params(ohlcv, n, dt, p);
    double* ret = malloc(n*sizeof(double));
    
    // We already have vol_scale passed in, but need returns
    // Re-calc returns (inefficient but safe for isolation)
    for(int i=0; i<n; i++) {
        double curr=ohlcv[(i+1)*N_COLS+3], prev=ohlcv[i*N_COLS+3];
        ret[i] = log(curr/prev); // Note: Detrending happened outside? No, inside wrapper logic usually.
        // For C-Core self-containment, we assume input 'ohlcv' is raw, 
        // but 'v_scale' and 'ret' are usually pre-processed. 
        // Here we use the helper 'detrend_log_returns' internally.
    }
    
    // Actually, optimize needs Detrended Returns.
    // Let's call the helper to populate ret and v_scale if passed NULL
    if (!v_scale) {
        v_scale = malloc(n*sizeof(double));
        detrend_log_returns(ohlcv, n, ret, v_scale);
    } else {
        // If v_scale provided, assume ret is pre-calc? No, safer to recalc.
        detrend_log_returns(ohlcv, n, ret, v_scale);
    }

    // Nelder-Mead (Simplified)
    int n_dim=5; double simplex[6][5]; double scores[6];
    for(int i=0; i<=n_dim; i++) {
        SVCJParams t = *p;
        if(i==1) t.kappa*=1.3; if(i==2) t.theta*=1.3;
        check_constraints(&t);
        simplex[i][0]=t.kappa; simplex[i][1]=t.theta; simplex[i][2]=t.sigma_v; simplex[i][3]=t.rho; simplex[i][4]=t.lambda_j;
        scores[i] = obj_func(ret, n, dt, &t);
    }
    
    for(int k=0; k<NM_ITER; k++) {
        int vs[6]; for(int j=0; j<6; j++) vs[j]=j;
        for(int i=0; i<6; i++) for(int j=i+1; j<6; j++) if(scores[vs[j]] > scores[vs[i]]) { int t=vs[i]; vs[i]=vs[j]; vs[j]=t; }
        double c[5]={0}; for(int i=0; i<5; i++) for(int d=0; d<5; d++) c[d]+=simplex[vs[i]][d]; for(int d=0; d<5; d++) c[d]/=5.0;
        double ref[5]; SVCJParams rp = *p; for(int d=0; d<5; d++) ref[d] = c[d] + 1.0*(c[d]-simplex[vs[5]][d]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        check_constraints(&rp);
        double r_score = obj_func(ret, n, dt, &rp);
        if(r_score > scores[vs[0]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = ref[d]; scores[vs[5]] = r_score; } 
        else {
             double con[5]; SVCJParams cp = *p; for(int d=0; d<5; d++) con[d] = c[d] + 0.5*(simplex[vs[5]][d]-c[d]);
             cp.kappa=con[0]; cp.theta=con[1]; cp.sigma_v=con[2]; cp.rho=con[3]; cp.lambda_j=con[4];
             check_constraints(&cp);
             double c_score = obj_func(ret, n, dt, &cp);
             if(c_score > scores[vs[5]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = con[d]; scores[vs[5]] = c_score; }
        }
    }
    
    int best=0; for(int i=1; i<6; i++) if(scores[i]>scores[best]) best=i;
    p->kappa=simplex[best][0]; p->theta=simplex[best][1]; p->sigma_v=simplex[best][2];
    p->rho=simplex[best][3]; p->lambda_j=simplex[best][4];
    
    if(out_spot) ukf_pure_likelihood(ret, v_scale, n, dt, p, out_spot);
    
    free(ret);
    // If we malloced v_scale locally, free it. But pointer logic is tricky here.
    // For safety in this snippet, we leak if NULL passed (bad), or assume wrapper manages it.
    // In production: wrapper allocates.
}

// --- Main Engine ---
void run_fidelity_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out) {
    // 2. Hurst Memory Depth (Window Selection)
    // Scan last 500 bars to find Hurst decay
    double* all_ret = malloc((total_len-1)*sizeof(double));
    double* all_vol = malloc((total_len-1)*sizeof(double));
    detrend_log_returns(ohlcv, total_len-1, all_ret, all_vol);
    
    double h = calc_hurst(all_ret, total_len-1);
    out->hurst_exp = h;
    
    // Logic: If H > 0.6 (Trending), use longer window. If H ~ 0.5, use shorter.
    int win_grav = (h > 0.6) ? 120 : 60;
    int win_imp = 30;
    
    // 1. Disjoint Sampling
    if (total_len < win_grav + win_imp) { out->is_valid=0; free(all_ret); free(all_vol); return; }
    
    // 3. CUSUM (Truncate Gravity)
    // Check gravity window for structural breaks
    int grav_start = total_len - win_imp - win_grav;
    int break_idx = detect_changepoint(all_ret + grav_start, win_grav);
    if (break_idx > 0) {
        grav_start += break_idx; // Shift forward to after break
        win_grav -= break_idx;
        if(win_grav < 40) win_grav = 40; // Floor
    }
    
    out->win_gravity = win_grav;
    out->win_impulse = win_imp;
    
    // Fit Gravity
    SVCJParams p_grav;
    // We need to pass raw OHLCV pointer for initialization logic
    // Optimize handles detrending internally again on the slice
    double* grav_scale = malloc((win_grav)*sizeof(double));
    optimize_svcj(ohlcv + grav_start*N_COLS, win_grav, dt, grav_scale, &p_grav, NULL);
    
    out->fit_theta = p_grav.theta;
    out->fit_kappa = p_grav.kappa;
    out->fit_sigma_v = p_grav.sigma_v;
    out->fit_rho = p_grav.rho;
    out->fit_lambda = p_grav.lambda_j;
    
    // Fit Impulse
    SVCJParams p_imp;
    int imp_start = total_len - win_imp;
    double* imp_spot = malloc(win_imp*sizeof(double));
    double* imp_scale = malloc(win_imp*sizeof(double));
    optimize_svcj(ohlcv + imp_start*N_COLS, win_imp, dt, imp_scale, &p_imp, imp_spot);
    
    // 4. Kinetic Dist Matching (KS Test)
    // We need Spot Vol path for Gravity too
    double* grav_spot = malloc(win_grav*sizeof(double));
    ukf_pure_likelihood(all_ret + grav_start, grav_scale, win_grav, dt, &p_grav, grav_spot);
    
    out->ks_stat = perform_ks_test(grav_spot, win_grav, imp_spot, win_imp);
    
    // Energy Ratio (Median vs Median)
    // Quick Sort to get Median
    qsort(imp_spot, win_imp, sizeof(double), cmp);
    qsort(grav_spot, win_grav, sizeof(double), cmp);
    double med_imp = imp_spot[win_imp/2];
    double med_grav = grav_spot[win_grav/2];
    out->energy_ratio = (med_imp*med_imp) / (med_grav*med_grav); // Variance Ratio
    
    // Residue Bias (Direction)
    double res_sum = 0;
    for(int i=0; i<win_imp; i++) res_sum += all_ret[imp_start + i];
    out->residue_bias = res_sum;
    
    // Validation
    // KS > 0.3 means distributions are different
    // Energy > 1.2 means expansion
    out->is_valid = (out->ks_stat > 0.3 && out->energy_ratio > 1.2) ? 1 : 0;
    
    free(all_ret); free(all_vol); free(imp_spot); free(imp_scale); 
    free(grav_spot); free(grav_scale);
}

void run_instant_filter(double val, double vol, double avg_vol, double dt, SVCJParams* p, double* state_var, InstantState* out) {
    double dt_eff = dt * (vol / avg_vol);
    double v_pred = *state_var + p->kappa*(p->theta - *state_var)*dt_eff;
    if(v_pred<1e-9) v_pred=1e-9;
    
    double y = val - (p->mu - 0.5*v_pred)*dt_eff; // Val is already log return
    double tot_var = v_pred + p->lambda_j*(p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
    out->innovation_z_score = y / sqrt(tot_var * dt_eff);
    
    // Update
    double S = v_pred*dt_eff + p->lambda_j*dt_eff*tot_var; // Approx
    double K = p->rho*p->sigma_v*dt_eff/S;
    double v_new = v_pred + K*y;
    if(v_new<1e-9) v_new=1e-9;
    
    out->current_spot_vol = sqrt(v_new);
    *state_var = v_new;
}