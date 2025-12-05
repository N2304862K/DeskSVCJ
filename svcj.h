#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 300
#define SQRT_2PI 2.50662827463
#define N_COLS 5

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    double current_spot_vol;
    double current_jump_prob;
    double innovation_z_score;
} InstantState;

typedef struct {
    double energy_ratio;
    double residue_bias;
    double f_p_value;
    double t_p_value;
    int is_valid;
    int win_impulse;
    int win_gravity;
} FidelityMetrics;

// Core
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

// Optimization
void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);

// Fidelity Engine
void run_fidelity_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out);

// Instant Filter
void run_instant_filter(double return_val, double dt, SVCJParams* p, double* state_var, InstantState* out);

#endif