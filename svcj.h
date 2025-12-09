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
    double jerk; // d(Z_score)/dt (Acceleration)
} InstantState;

typedef struct {
    // Signals
    double energy_ratio;
    double residue_bias;
    
    // Stats
    double levene_p;     // Volatility Significance
    double mw_p;         // Drift Significance
    int is_valid;
    
    // Physics
    double fit_theta, fit_kappa, fit_sigma_v, fit_rho, fit_lambda;
} FidelityMetrics;

// Core
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);
void compute_detrended_returns(double* ohlcv, int n_rows, double* out_returns);
double get_avg_volume(double* ohlcv, int n);

// Sort
void sort_doubles_fast(double* arr, int n);

// Engines
void run_full_audit_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out);
void run_instant_filter_vol(double ret, double vol, double avg_vol, double dt, SVCJParams* p, double* state, double* prev_z, InstantState* out);

#endif