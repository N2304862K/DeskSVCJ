#include "svcj.h"
#include <float.h>

// =======================================================
// 1. INTERNAL STATE & FILE I/O
// =======================================================

// These variables are PERSISTENT inside the compiled .so/.dll
static SVCJParams g_params;
static double g_dt;
static double g_state_variance;
static double g_avg_volume;
static double g_tick_buffer[TICK_BUFFER_SIZE];
static int g_tick_idx = 0;
static double g_last_z = 0;
// Model Health
static double g_log_likelihood_sum = 0;
static double g_baseline_ll_per_tick = 0;
static int g_ticks_since_calib = 0;

int load_physics(const char* ticker, SVCJParams* p) {
    char filename[256];
    sprintf(filename, "./%s.bin", ticker);
    FILE* f = fopen(filename, "rb");
    if(!f) return 0;
    
    fread(p, sizeof(SVCJParams), 1, f);
    fclose(f);
    return 1;
}

void save_physics(const char* ticker, SVCJParams* p) {
    char filename[256];
    sprintf(filename, "./%s.bin", ticker);
    FILE* f = fopen(filename, "wb");
    if(!f) return;
    
    fwrite(p, sizeof(SVCJParams), 1, f);
    fclose(f);
}

// =======================================================
// 2. CORE OPTIMIZATION (Condensed)
// =======================================================
void check_constraints(SVCJParams* p) {
    if(p->theta<1e-6)p->theta=1e-6; if(p->kappa<0.1)p->kappa=0.1;
    if(p->sigma_v<0.01)p->sigma_v=0.01; if(p->lambda_j<0.01)p->lambda_j=0.01;
}

double ukf_vol_ll(double* r, double* v, int n, double dt, double av, SVCJParams* p) {
    double ll=0; double var_state=p->theta; 
    for(int t=0; t<n; t++) {
        double dt_eff = dt * (v[t]/av);
        double v_pred = var_state + p->kappa*(p->theta - var_state)*dt_eff;
        if(v_pred<1e-9)v_pred=1e-9;
        double y = r[t] - (p->mu - 0.5*v_pred)*dt_eff;
        double S = v_pred*dt_eff + p->lambda_j*dt_eff*(p->mu_j*p->mu_j+p->sigma_j*p->sigma_j);
        if(S<1e-9)S=1e-9;
        double pdf = (1.0/sqrt(2*M_PI*S))*exp(-0.5*y*y/S);
        ll += log(pdf + 1e-20);
        var_state = v_pred + 0.1*(y*y - S);
        if(var_state<1e-9)var_state=1e-9;
    }
    return ll;
}

void optimize_svcj_core(double* returns, double* volumes, int n, double dt, SVCJParams* p) {
    // Basic init
    p->theta=0.04; p->kappa=3.0; p->sigma_v=0.5; p->rho=-0.5; p->lambda_j=0.5;
    
    double avg_vol = 0; for(int i=0; i<n; i++) avg_vol += volumes[i]; avg_vol /= n;
    
    // Condensed Nelder-Mead
    // In full impl, use full loop. Here we just set params based on rough estimate.
    double sum_sq=0; for(int i=0;i<n;i++) sum_sq+=returns[i]*returns[i];
    p->theta = (sum_sq/n)/dt;
    check_constraints(p);
}

void compute_log_returns(double* ohlcv, int n, double* out_r, double* out_v) {
    for(int i=1; i<n; i++) {
        out_r[i-1] = log(ohlcv[i*N_COLS+3]/ohlcv[(i-1)*N_COLS+3]);
        out_v[i-1] = ohlcv[i*N_COLS+4];
    }
}

// =======================================================
// 3. THE HIGH-SPEED ENGINE
// =======================================================

void initialize_tick_engine(SVCJParams* p, double dt) {
    g_params = *p;
    g_dt = dt;
    g_state_variance = p->theta;
    g_tick_idx = 0;
    g_last_z = 0;
    g_log_likelihood_sum = 0;
    g_ticks_since_calib = 0;
    
    // Calculate Baseline Likelihood
    // A normal move (Z=0) has this likelihood. We detect deviations from this.
    double var_exp = (g_params.theta + g_params.lambda_j*(g_params.mu_j*g_params.mu_j+g_params.sigma_j*g_params.sigma_j))*g_dt;
    g_baseline_ll_per_tick = -0.5*log(2*M_PI*var_exp);
}

void run_tick_update(double price, double volume, InstantMetrics* out) {
    // --- 1. State ---
    static double last_price = -1.0;
    if(last_price < 0) { last_price = price; out->z_score=0; out->jerk=0; return; }
    
    double ret = log(price / last_price);
    last_price = price;
    
    // --- 2. Volume Clock & Prediction ---
    // Note: avg_volume needs to be set by calibration, here we assume it's stable-ish
    double time_scale = 1.0; // In prod, this uses a rolling avg from calibration
    double dt_eff = g_dt * time_scale;
    
    double v_pred = g_state_variance + g_params.kappa*(g_params.theta - g_state_variance)*dt_eff;
    if(v_pred < 1e-9) v_pred = 1e-9;
    
    // --- 3. Z-Score (The Signal) ---
    double y = ret - (g_params.mu - 0.5*v_pred)*dt_eff;
    double jump_var = g_params.lambda_j * (g_params.mu_j*g_params.mu_j + g_params.sigma_j*g_params.sigma_j);
    double step_std = sqrt((v_pred + jump_var) * dt_eff);
    if(step_std < 1e-9) step_std = 1e-9;
    
    out->z_score = y / step_std;
    
    // --- 4. Jerk (First Derivative) ---
    out->jerk = out->z_score - g_last_z;
    g_last_z = out->z_score;
    
    // --- 5. Skew (Second Derivative) ---
    g_tick_buffer[g_tick_idx % TICK_BUFFER_SIZE] = out->z_score;
    g_tick_idx++;
    
    if(g_tick_idx >= TICK_BUFFER_SIZE) {
        double mean=0;
        for(int i=0; i<TICK_BUFFER_SIZE; i++) mean += g_tick_buffer[i];
        mean /= TICK_BUFFER_SIZE;
        
        double m2=0, m3=0;
        for(int i=0; i<TICK_BUFFER_SIZE; i++) {
            double d = g_tick_buffer[i] - mean;
            m2 += d*d; m3 += d*d*d;
        }
        m2 /= TICK_BUFFER_SIZE;
        m3 /= TICK_BUFFER_SIZE;
        
        out->skew = (m2 > 1e-9) ? m3 / pow(m2, 1.5) : 0;
    } else {
        out->skew = 0;
    }
    
    // --- 6. State Update ---
    double S = v_pred*dt_eff + jump_var*dt_eff;
    double K = (g_params.rho * g_params.sigma_v * dt_eff)/S;
    g_state_variance = v_pred + K * y;
    if(g_state_variance < 1e-9) g_state_variance = 1e-9;
    
    // --- 7. Model Failure Detection ---
    double current_ll = -0.5 * log(2*M_PI*S) - 0.5*y*y/S;
    g_log_likelihood_sum += current_ll;
    g_ticks_since_calib++;
    
    // Chow Test Analogy: Is the average likelihood of this window
    // significantly worse than the baseline expectation?
    if (g_ticks_since_calib > 50) { // Wait for buffer
        double avg_ll = g_log_likelihood_sum / g_ticks_since_calib;
        
        // If our model is consistently worse than the "Normal" model
        // by a significant margin (e.g. 2x in log-space), it has failed.
        if (avg_ll < g_baseline_ll_per_tick * 2.0) {
            out->recalibrate_flag = 1;
        } else {
            out->recalibrate_flag = 0;
        }
    } else {
        out->recalibrate_flag = 0;
    }
}

// =======================================================
// 4. THE SELF-ORGANIZING CALIBRATOR
// =======================================================
void run_full_calibration(const char* ticker, double* ohlcv, int n, double dt) {
    // 1. Prepare Data
    double* ret = malloc((n-1)*sizeof(double));
    double* vol = malloc((n-1)*sizeof(double));
    compute_log_returns(ohlcv, n, ret, vol);
    
    // 2. Optimize
    SVCJParams new_params;
    optimize_svcj_core(ret, vol, n-1, dt, &new_params);
    
    // 3. Save to Disk
    save_physics(ticker, &new_params);
    
    // 4. Reload into Live State
    initialize_tick_engine(&new_params, dt);
    
    free(ret); free(vol);
}