#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 250
#define N_COLS 5

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    int window;
    double sigma_v;      // The Stability Metric
    double theta;        // The Structural Vol
} VoVPoint;

typedef struct {
    // Physics
    int win_impulse;
    int win_gravity;
    double energy_ratio;
    double residue_bias;
    
    // Statistical Tests
    double f_p_value;    // Energy Significance
    double t_p_value;    // Direction Significance
    double lb_p_value;   // Autocorrelation Significance (Momentum Validity)
    double lb_stat;
    
    int is_valid;
} FidelityMetrics;

// Utils
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

// Core
void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);

// New Engines
void run_vov_scan(double* ohlcv, int total_len, double dt, int step, VoVPoint* out_buffer, int max_steps);
void run_fidelity_check(double* ohlcv, int total_len, int win_grav, int win_imp, double dt, FidelityMetrics* out);

#endif