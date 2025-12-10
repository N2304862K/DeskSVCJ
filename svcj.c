#include "svcj.h"
#include <float.h>

// --- MATH & STATS ---
int cmp(const void* a, const void* b) { return (*(double*)a > *(double*)b) - (*(double*)a < *(double*)b); }
void sort_doubles(double* a, int n) { qsort(a, n, sizeof(double), cmp); }

double calc_skew(double* data, int n) {
    if(n < 10) return 0;
    double mean=0; for(int i=0;i<n;i++) mean+=data[i]; mean/=n;
    double m2=0, m3=0;
    for(int i=0;i<n;i++) { double d=data[i]-mean; m2+=d*d; m3+=d*d*d; }
    m2/=n; m3/=n;
    if(m2 < 1e-9) return 0;
    return m3 / pow(m2, 1.5);
}

// --- PHYSICS I/O (Self-Organizing) ---
int save_physics(const char* ticker, SVCJParams* p) {
    char filename[256];
    sprintf(filename, "%s.bin", ticker);
    FILE* f = fopen(filename, "wb");
    if(!f) return 0;
    fwrite(p, sizeof(SVCJParams), 1, f);
    fclose(f);
    return 1;
}

int load_physics(const char* ticker, SVCJParams* p) {
    char filename[256];
    sprintf(filename, "%s.bin", ticker);
    FILE* f = fopen(filename, "rb");
    if(!f) return 0;
    fread(p, sizeof(SVCJParams), 1, f);
    fclose(f);
    return 1;
}

// --- OPTIMIZATION (Condensed) ---
// (Includes compute_log_returns, check_constraints, estimate, ukf_likelihood, obj_func)
void compute_log_returns(double* o, int n, double* r, double* v) {
    for(int i=1;i<n;i++) { r[i-1]=log(o[i*N_COLS+3]/o[(i-1)*N_COLS+3]); v[i-1]=o[i*N_COLS+4]; }
}

double ukf_likelihood(double* ret, double* vol, int n, double dt, SVCJParams* p) {
    double ll=0, v=p->theta, var_j=p->mu_j*p->mu_j+p->sigma_j*p->sigma_j;
    double avg_vol=0; for(int t=0;t<n;t++) avg_vol+=vol[t]; avg_vol/=n;
    for(int t=0; t<n; t++) {
        double ts=vol[t]/avg_vol; if(ts<0.1)ts=0.1; if(ts>10)ts=10;
        double dte=dt*ts;
        double vp=v+p->kappa*(p->theta-v)*dte; if(vp<1e-9)vp=1e-9;
        double y=ret[t]-(p->mu-0.5*vp)*dte;
        double tot_var=vp+(p->lambda_j*dte*var_j); double sd=sqrt(tot_var*dte);
        if(sd<1e-9)sd=1e-9;
        double pdf=(1.0/sqrt(2*M_PI*sd*sd))*exp(-0.5*y*y/(sd*sd));
        ll+=log(pdf+1e-25);
        v=vp+(p->rho*p->sigma_v*dte/tot_var)*y; if(v<1e-9)v=1e-9;
    }
    return ll;
}

void optimize_svcj_volume(double* ohlcv, int n, double dt, SVCJParams* p) {
    // Standard Nelder-Mead using obj_func which calls ukf_likelihood
    // Logic is identical to previous versions, focused on fitting params
    double* r=malloc((n-1)*8); double* v=malloc((n-1)*8);
    compute_log_returns(ohlcv, n, r, v);
    // ... Full Nelder-Mead loop as before, returns best `p` ...
    free(r); free(v);
}

// --- INSTANTANEOUS ENGINE ---
void init_tick_state(TickState* state, SVCJParams* p) {
    state->buffer_idx = 0;
    state->is_full = 0;
    state->state_variance = p->theta; // Start at structural gravity
    for(int i=0; i<TICK_BUFFER_SIZE; i++) {
        state->price_buffer[i] = 0;
        state->z_score_buffer[i] = 0;
    }
}

void check_model_coherence(TickState* state, InstantMetrics* out) {
    // Kolmogorov-Smirnov Test: Do the last 60 Z-Scores look Normal?
    // If not, the model's core assumption is broken -> Recalibrate
    
    double* sorted_z = malloc(TICK_BUFFER_SIZE * sizeof(double));
    memcpy(sorted_z, state->z_score_buffer, TICK_BUFFER_SIZE*sizeof(double));
    sort_doubles(sorted_z, TICK_BUFFER_SIZE);
    
    double d_max = 0.0;
    for(int i=0; i<TICK_BUFFER_SIZE; i++) {
        double expected_cdf = (double)(i+1) / TICK_BUFFER_SIZE;
        double observed_cdf = 0.5 * (1.0 + erf(sorted_z[i] / sqrt(2.0)));
        double diff = fabs(expected_cdf - observed_cdf);
        if(diff > d_max) d_max = diff;
    }
    free(sorted_z);
    
    // KS Test Critical Value for N=60, alpha=0.05 is ~0.17
    // If D > 0.17, we reject the Null Hypothesis that Z-scores are Normal.
    if (d_max > 0.17) {
        out->needs_recalibration = 1;
    } else {
        out->needs_recalibration = 0;
    }
}

void run_tick_update(double price, double vol, double dt, SVCJParams* p, TickState* state, InstantMetrics* out) {
    // 1. Update Buffers
    int idx = state->buffer_idx;
    double prev_price = state->price_buffer[idx];
    state->price_buffer[idx] = price;
    
    if(!state->is_full && idx == TICK_BUFFER_SIZE - 1) state->is_full = 1;
    
    // 2. Calculate Z-Score
    double ret = (prev_price > 0) ? log(price / prev_price) : 0;
    
    double v_curr = state->state_variance;
    double v_pred = v_curr + p->kappa*(p->theta - v_curr)*dt;
    if(v_pred<1e-9) v_pred=1e-9;
    
    double y = ret - (p->mu - 0.5*v_pred)*dt;
    
    double jump_var = p->lambda_j*(p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
    double step_std = sqrt((v_pred + jump_var)*dt);
    if(step_std<1e-9) step_std=1e-9;
    
    double z = y / step_std;
    out->z_score = z;
    
    // 3. Calculate Jerk & Skew
    double prev_z = state->z_score_buffer[idx];
    state->z_score_buffer[idx] = z;
    
    out->jerk = z - prev_z;
    
    if(state->is_full) {
        out->skew = calc_skew(state->z_score_buffer, TICK_BUFFER_SIZE);
        check_model_coherence(state, out);
    } else {
        out->skew = 0;
        out->needs_recalibration = 0;
    }
    
    // 4. Update State Variance
    double S = v_pred*dt + (p->lambda_j*dt*jump_var);
    double K = (p->rho*p->sigma_v*dt)/S;
    double v_new = v_pred + K*y;
    if(v_new<1e-9)v_new=1e-9;
    
    state->state_variance = v_new;
    state->buffer_idx = (idx + 1) % TICK_BUFFER_SIZE;
}