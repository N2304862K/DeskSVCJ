#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 300
#define N_COLS 5 // Open, High, Low, Close, Volume

// 1. The Physics (Parameters)
typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

// 2. The State (Instantaneous)
typedef struct {
    double current_spot_vol;
    double current_jump_prob;
    double innovation_z_score;
    double sprt_score; // Sequential Probability Ratio Test (Drift)
} InstantState;

// 3. The Fidelity Report (Statistical Validation)
typedef struct {
    // Physicals
    int win_impulse;
    int win_gravity;
    double energy_ratio;
    double residue_bias;
    
    // Stats
    double ks_stat;      // Kinetic Distribution Match
    double hurst_exp;    // Memory Depth
    
    int is_valid;        // 1 = Pass, 0 = Fail
    
    // Physics Export (To sync Monitor)
    double fit_theta;
    double fit_kappa;
    double fit_sigma_v;
    double fit_rho;
    double fit_lambda;
} FidelityMetrics;

// 4. Utils & Core
void detrend_log_returns(double* ohlcv, int n, double* out_ret, double* out_vol_scale);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);

// 5. Engines
void run_fidelity_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out);
void run_instant_filter(double return_val, double dt, double vol_scale, SVCJParams* p, double* state_var, double* sprt_accum, InstantState* out);

#endif