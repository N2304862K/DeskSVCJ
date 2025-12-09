#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 300
#define N_COLS 5 // Open,High,Low,Close,Volume

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    double current_spot_vol;
    double current_jump_prob;
    double innovation_z_score;
} InstantState;

typedef struct {
    int win_impulse, win_gravity;
    double energy_ratio;
    double residue_bias;
    
    // Advanced Stats
    double ks_stat;      // Kinetic Distribution Match
    double hurst_exp;    // Memory Depth
    double vol_scaling;  // Volume Time Factor
    
    int is_valid;
    
    // Physics Payload
    double fit_theta, fit_kappa, fit_sigma_v, fit_rho, fit_lambda;
} FidelityMetrics;

// --- Core ---
void detrend_log_returns(double* ohlcv, int n, double* out_ret, double* out_vol_scale);
void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p);
void optimize_svcj(double* ohlcv, int n, double dt, double* vol_scale, SVCJParams* p, double* out_spot);

// --- Advanced Engines ---
double calc_hurst(double* returns, int n);
int detect_changepoint(double* returns, int n);
double perform_ks_test(double* g1, int n1, double* g2, int n2);

void run_fidelity_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out);
void run_instant_filter(double val, double vol, double avg_vol, double dt, SVCJParams* p, double* state, InstantState* out);

#endif