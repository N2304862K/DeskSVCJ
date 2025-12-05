#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 300
#define N_COLS 5
#define MIN_FREQ_SEP 4.0 // Gravity must be 4x longer than Impulse

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    // Physicals
    int win_impulse;
    int win_gravity;
    double energy_ratio;
    double residue_bias;
    
    // Fidelity Stats
    double f_stat;       // Variance Ratio Statistic
    double f_p_value;    // Significance of Vol Expansion
    double t_stat;       // Drift Statistic
    double t_p_value;    // Significance of Direction
    
    int is_valid;        // 1 if Statistical Tests Pass
} FidelityMetrics;

// Utils
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

// Core
void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);

// The New Engine
void run_fidelity_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out);

#endif