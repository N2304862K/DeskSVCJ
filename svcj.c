#include "svcj.h"
#include <float.h>

// --- SORTING HELPERS ---
int compare_doubles(const void* a, const void* b) {
    double arg1 = *(const double*)a;
    double arg2 = *(const double*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

// --- STATISTICAL FUNCTIONS ---
double norm_cdf(double x) {
    double t = 1.0 / (1.0 + 0.5 * fabs(x));
    double tau = t * exp(-x*x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? tau : 2.0 - tau;
}

// 1. Median Calculator (Requires Sort)
double calc_median(double* data, int n) {
    double* sorted = malloc(n * sizeof(double));
    memcpy(sorted, data, n * sizeof(double));
    qsort(sorted, n, sizeof(double), compare_doubles);
    double med = (n % 2 == 0) ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0 : sorted[n/2];
    free(sorted);
    return med;
}

// 2. Levene's Test (Brown-Forsythe: Robust F-Test)
// Tests equality of variances using absolute deviation from median
double perform_levene_test(double* group1, int n1, double* group2, int n2) {
    double med1 = calc_median(group1, n1);
    double med2 = calc_median(group2, n2);
    
    // Transform to absolute deviations (Z)
    double* z1 = malloc(n1 * sizeof(double));
    double* z2 = malloc(n2 * sizeof(double));
    double sum_z1=0, sum_z2=0;
    
    for(int i=0; i<n1; i++) { z1[i] = fabs(group1[i] - med1); sum_z1 += z1[i]; }
    for(int i=0; i<n2; i++) { z2[i] = fabs(group2[i] - med2); sum_z2 += z2[i]; }
    
    double mean_z1 = sum_z1/n1;
    double mean_z2 = sum_z2/n2;
    double grand_mean = (sum_z1 + sum_z2) / (n1 + n2);
    
    // ANOVA on Z
    double ssb = n1*(mean_z1-grand_mean)*(mean_z1-grand_mean) + n2*(mean_z2-grand_mean)*(mean_z2-grand_mean);
    double ssw = 0;
    for(int i=0; i<n1; i++) ssw += (z1[i]-mean_z1)*(z1[i]-mean_z1);
    for(int i=0; i<n2; i++) ssw += (z2[i]-mean_z2)*(z2[i]-mean_z2);
    
    free(z1); free(z2);
    
    if (ssw < 1e-9) return 1.0; // Identical
    double f = (ssb / 1.0) / (ssw / (n1 + n2 - 2));
    
    // F-Test P-Value (Approx)
    // Degrees of Freedom: (1, N-2)
    // Use Normal Approx for simplicity in C
    return 2.0 * norm_cdf(-sqrt(f)); // Rough p-value
}

// 3. Mann-Whitney U Test (Robust T-Test)
// Tests shift in location (Median difference)
double perform_mann_whitney(double* group1, int n1, double* group2, int n2) {
    int total_n = n1 + n2;
    
    // Create combined rank array
    typedef struct { double val; int group; double rank; } RankItem;
    RankItem* items = malloc(total_n * sizeof(RankItem));
    
    for(int i=0; i<n1; i++) { items[i].val = group1[i]; items[i].group = 1; }
    for(int i=0; i<n2; i++) { items[n1+i].val = group2[i]; items[n1+i].group = 2; }
    
    // Sort by value to assign ranks
    // Need custom comparator for struct
    int compare_ranks(const void* a, const void* b) {
        double v1 = ((RankItem*)a)->val;
        double v2 = ((RankItem*)b)->val;
        return (v1 > v2) - (v1 < v2);
    }
    qsort(items, total_n, sizeof(RankItem), compare_ranks);
    
    // Assign Ranks (Handle ties)
    for(int i=0; i<total_n; ) {
        int j = i;
        while(j < total_n && items[j].val == items[i].val) j++;
        double rank = (i + 1 + j) / 2.0;
        for(int k=i; k<j; k++) items[k].rank = rank;
        i = j;
    }
    
    // Sum Ranks for Group 1
    double r1 = 0;
    for(int i=0; i<total_n; i++) {
        if(items[i].group == 1) r1 += items[i].rank;
    }
    
    free(items);
    
    // U Statistic
    double u1 = r1 - (n1*(n1+1))/2.0;
    double u2 = (n1*n2) - u1;
    double u = (u1 < u2) ? u1 : u2; // Minimum U
    
    // Z-Score approximation
    double mu_u = (n1*n2)/2.0;
    double sigma_u = sqrt((n1*n2*(n1+n2+1))/12.0);
    double z = (u - mu_u) / sigma_u;
    
    return 2.0 * norm_cdf(-fabs(z)); // P-Value
}

// 4. Kolmogorov-Smirnov Test (Regime Shape)
double perform_ks_test(double* group1, int n1, double* group2, int n2) {
    double* s1 = malloc(n1 * sizeof(double)); memcpy(s1, group1, n1*sizeof(double));
    double* s2 = malloc(n2 * sizeof(double)); memcpy(s2, group2, n2*sizeof(double));
    qsort(s1, n1, sizeof(double), compare_doubles);
    qsort(s2, n2, sizeof(double), compare_doubles);
    
    double max_d = 0.0;
    int i=0, j=0;
    while(i < n1 && j < n2) {
        double d1 = s1[i];
        double d2 = s2[j];
        double cdf1 = (double)(i) / n1;
        double cdf2 = (double)(j) / n2;
        
        double diff = fabs(cdf1 - cdf2);
        if (diff > max_d) max_d = diff;
        
        if (d1 <= d2) i++;
        else j++;
    }
    
    free(s1); free(s2);
    
    // Kolmogorov distribution approx
    double ne = (double)(n1*n2)/(n1+n2);
    double lambda = (sqrt(ne) + 0.12 + 0.11/sqrt(ne)) * max_d;
    // P-Value approx: 2 * sum (-1)^(k-1) * exp(-2*k^2*lambda^2)
    double p = 0.0;
    for(int k=1; k<=5; k++) {
        p += 2 * pow(-1, k-1) * exp(-2*k*k*lambda*lambda);
    }
    if (p < 0) p = 0; if (p > 1) p = 1;
    return p; // KS P-Value: Small means distributions are DIFFERENT
}

// --- SVCJ CORE (Standard) ---
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) {
    for(int i=1; i<n_rows; i++) {
        double prev = ohlcv[(i-1)*N_COLS+3]; double curr = ohlcv[i*N_COLS+3];
        if(prev < 1e-9) prev = 1e-9;
        out_returns[i-1] = log(curr/prev);
    }
}

void check_constraints(SVCJParams* p) {
    if(p->kappa<0.01) p->kappa=0.01; if(p->kappa>100.0) p->kappa=100.0;
    if(p->theta<1e-6) p->theta=1e-6; if(p->theta>50.0) p->theta=50.0;
    if(p->sigma_v<0.01) p->sigma_v=0.01; if(p->sigma_v>50.0) p->sigma_v=50.0;
    if(p->rho>0.999) p->rho=0.999; if(p->rho<-0.999) p->rho=-0.999;
    if(p->lambda_j<0.001) p->lambda_j=0.001; if(p->lambda_j>3000.0) p->lambda_j=3000.0;
    if(p->sigma_j<0.0001) p->sigma_j=0.0001;
    double feller = 2.0*p->kappa*p->theta;
    if(p->sigma_v*p->sigma_v > feller*20.0) p->sigma_v=sqrt(feller*20.0);
}

void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p) {
    double sum_gk=0;
    for(int i=0; i<n; i++) {
        double O=ohlcv[i*N_COLS]; double H=ohlcv[i*N_COLS+1];
        double L=ohlcv[i*N_COLS+2]; double C=ohlcv[i*N_COLS+3];
        if(L<1e-9)L=1e-9; if(O<1e-9)O=1e-9;
        double hl=log(H/L); double co=log(C/O);
        double val=0.5*hl*hl - (2.0*log(2.0)-1.0)*co*co;
        if(val>0) sum_gk+=val;
    }
    double rv=(sum_gk/n)/dt;
    p->mu=0; p->theta=rv; p->kappa=2.0; p->sigma_v=sqrt(rv);
    p->rho=-0.5; p->lambda_j=0.5; p->mu_j=0; p->sigma_j=sqrt(rv);
    check_constraints(p);
}

double ukf_pure_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double ll=0; double v=p->theta; double var_j=p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    for(int t=0; t<n; t++) {
        double v_pred = v + p->kappa*(p->theta - v)*dt;
        if(v_pred<1e-9) v_pred=1e-9;
        double y = returns[t] - (p->mu - 0.5*v_pred)*dt;
        double rob_var = (v_pred<1e-9)?1e-9:v_pred; rob_var*=dt;
        double pdf_d = (1.0/sqrt(rob_var*2*M_PI))*exp(-0.5*y*y/rob_var);
        double tot_j = rob_var+var_j; double yj=y-p->mu_j;
        double pdf_j = (1.0/sqrt(tot_j*2*M_PI))*exp(-0.5*yj*yj/tot_j);
        double prior = p->lambda_j*dt; if(prior>0.999) prior=0.999;
        double den = pdf_j*prior + pdf_d*(1.0-prior);
        if(den<1e-25) den=1e-25;
        double post = (pdf_j*prior)/den;
        double S = v_pred*dt + post*var_j;
        v = v_pred + (p->rho*p->sigma_v*dt/S)*y;
        if(v<1e-9)v=1e-9; if(v>50.0)v=50.0;
        if(out_spot_vol) out_spot_vol[t]=sqrt(v_pred);
        if(out_jump_prob) out_jump_prob[t]=post;
        ll+=log(den);
    }
    return ll;
}

double obj_func(double* returns, int n, double dt, SVCJParams* p) {
    return ukf_pure_likelihood(returns, n, dt, p, NULL, NULL) - 0.05*(p->sigma_v*p->sigma_v);
}

void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    estimate_initial_params(ohlcv, n, dt, p);
    double* ret = malloc((n-1)*sizeof(double));
    if(!ret) return;
    compute_log_returns(ohlcv, n, ret);
    int n_dim=5; double simplex[6][5]; double scores[6];
    for(int i=0; i<=n_dim; i++) {
        SVCJParams t = *p;
        if(i==1) t.kappa*=1.3; if(i==2) t.theta*=1.3; if(i==3) t.sigma_v*=1.3;
        if(i==4) t.rho+=0.2;   if(i==5) t.lambda_j*=1.5;
        check_constraints(&t);
        simplex[i][0]=t.kappa; simplex[i][1]=t.theta; simplex[i][2]=t.sigma_v;
        simplex[i][3]=t.rho;   simplex[i][4]=t.lambda_j;
        scores[i] = obj_func(ret, n-1, dt, &t);
    }
    for(int k=0; k<NM_ITER; k++) {
        int vs[6]; for(int j=0; j<6; j++) vs[j]=j;
        for(int i=0; i<6; i++) { for(int j=i+1; j<6; j++) { if(scores[vs[j]] > scores[vs[i]]) { int tmp=vs[i]; vs[i]=vs[j]; vs[j]=tmp; } } }
        double c[5]={0}; for(int i=0; i<5; i++) { for(int d=0; d<5; d++) c[d]+=simplex[vs[i]][d]; } for(int d=0; d<5; d++) c[d]/=5.0;
        double ref[5]; SVCJParams rp = *p; for(int d=0; d<5; d++) ref[d] = c[d] + 1.0*(c[d]-simplex[vs[5]][d]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        check_constraints(&rp);
        double r_score = obj_func(ret, n-1, dt, &rp);
        if(r_score > scores[vs[0]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = ref[d]; scores[vs[5]] = r_score; } 
        else {
             double con[5]; SVCJParams cp = *p; for(int d=0; d<5; d++) con[d] = c[d] + 0.5*(simplex[vs[5]][d]-c[d]);
             cp.kappa=con[0]; cp.theta=con[1]; cp.sigma_v=con[2]; cp.rho=con[3]; cp.lambda_j=con[4];
             check_constraints(&cp);
             double c_score = obj_func(ret, n-1, dt, &cp);
             if(c_score > scores[vs[5]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = con[d]; scores[vs[5]] = c_score; }
        }
    }
    int best=0; for(int i=1; i<6; i++) if(scores[i]>scores[best]) best=i;
    p->kappa=simplex[best][0]; p->theta=simplex[best][1]; p->sigma_v=simplex[best][2];
    p->rho=simplex[best][3];   p->lambda_j=simplex[best][4];
    if(out_spot_vol) ukf_pure_likelihood(ret, n-1, dt, p, out_spot_vol, out_jump_prob);
    free(ret);
}

// --- NON-PARAMETRIC ENGINE ---
void run_nonparametric_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out) {
    int win_imp = 30; int win_grav = win_imp * 4;
    if (total_len < win_grav) { out->is_valid=0; return; }
    
    // 1. Gravity (Long)
    SVCJParams p_grav;
    int start_grav = total_len - win_grav;
    optimize_svcj(ohlcv + start_grav*N_COLS, win_grav, dt, &p_grav, NULL, NULL);
    
    // Export Physics
    out->fit_theta = p_grav.theta;
    out->fit_kappa = p_grav.kappa;
    out->fit_sigma_v = p_grav.sigma_v;
    out->fit_rho = p_grav.rho;
    out->fit_lambda = p_grav.lambda_j;
    
    // 2. Impulse (Short)
    SVCJParams p_imp;
    int start_imp = total_len - win_imp;
    double* imp_spot = malloc((win_imp-1)*sizeof(double));
    optimize_svcj(ohlcv + start_imp*N_COLS, win_imp, dt, &p_imp, imp_spot, NULL);
    out->energy_ratio = (imp_spot[win_imp-2]*imp_spot[win_imp-2]) / p_grav.theta;
    
    // 3. Prepare Data for Non-Parametric Tests
    double* ret_long = malloc((win_grav-1)*sizeof(double));
    compute_log_returns(ohlcv + start_grav*N_COLS, win_grav, ret_long);
    
    // Slice out the Impulse part of the returns
    double* ret_short = malloc((win_imp-1)*sizeof(double));
    compute_log_returns(ohlcv + start_imp*N_COLS, win_imp, ret_short);
    
    // 4. Robust Tests
    // Levene's (Energy Expansion)
    out->levene_p = perform_levene_test(ret_short, win_imp-1, ret_long, win_grav-1);
    
    // Mann-Whitney (Directional Shift)
    out->mw_p = perform_mann_whitney(ret_short, win_imp-1, ret_long, win_grav-1);
    
    // KS (Regime Shape)
    out->ks_p = perform_ks_test(ret_short, win_imp-1, ret_long, win_grav-1);
    
    // Median Residue (Direction)
    out->residue_bias = calc_median(ret_short, win_imp-1);
    
    out->win_impulse = win_imp;
    out->win_gravity = win_grav;
    
    // VALIDITY LOGIC (Robust)
    // 1. Levene < 0.05 (Variance Changed) OR KS < 0.05 (Shape Changed)
    // 2. MW < 0.10 (Direction Changed)
    int struct_break = (out->levene_p < 0.05 || out->ks_p < 0.05);
    int dir_break = (out->mw_p < 0.10);
    
    out->is_valid = (struct_break && dir_break) ? 1 : 0;
    
    free(imp_spot); free(ret_long); free(ret_short);
}

void run_instant_filter(double return_val, double dt, SVCJParams* p, double* state_var, InstantState* out) {
    double v_curr = *state_var;
    double v_pred = v_curr + p->kappa*(p->theta - v_curr)*dt;
    if(v_pred<1e-9) v_pred=1e-9;
    double y = return_val - (p->mu - 0.5*v_pred)*dt;
    double jump_var = p->lambda_j*(p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
    double total_std = sqrt((v_pred + jump_var)*dt);
    if(total_std<1e-9) total_std=1e-9;
    out->innovation_z_score = y / total_std;
    double S = v_pred*dt + (p->lambda_j*dt*jump_var);
    double K = (p->rho*p->sigma_v*dt)/S;
    double v_new = v_pred + K*y;
    if(v_new<1e-9) v_new=1e-9;
    out->current_spot_vol = sqrt(v_new);
    double pdf = (1.0/sqrt(2*M_PI*v_pred*dt))*exp(-0.5*y*y/(v_pred*dt));
    double pr = p->lambda_j*dt;
    out->current_jump_prob = pr / (pr + pdf*(1-pr));
    *state_var = v_new;
}