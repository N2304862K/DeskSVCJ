#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 300
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
    double f_p_value;    // Levene (Volume-Weighted)
    double t_p_value;    // Mann-Whitney (Raw)
    int is_valid;
    
    // Physics Export
    double fit_theta, fit_kappa, fit_sigma_v, fit_rho, fit_lambda;
} FidelityMetrics;

// Utils
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);
void compute_vol_weighted_returns(double* ohlcv, int n_rows, double avg_vol, double* out_returns);
double get_avg_volume(double* ohlcv, int n);

// Sort
void sort_doubles(double* arr, int n);

// Core
void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p);
void optimize_svcj_vol(double* ohlcv, int n, double dt, double avg_vol, SVCJParams* p, double* out_spot_vol);

// Engines
void run_full_audit_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out);
void run_instant_filter_vol(double ret, double vol, double avg_vol, double dt, SVCJParams* p, double* state, InstantState* out);

#endif