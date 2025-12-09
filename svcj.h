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
    // Windows
    int win_impulse;
    int win_gravity;
    
    // Physics
    double energy_ratio;
    double residue_median;
    
    // Non-Parametric Stats
    double levene_p;     // Energy (Robust Variance)
    double mw_p;         // Direction (Robust Drift)
    double ks_ret_p;     // Return Distribution Shape
    double ks_vol_p;     // Volatility Path Shape (New)
    double hurst_grav;   // Memory of Gravity Window
    
    // Output
    int is_valid;
    
    // Physics Payload
    double fit_theta;
    double fit_kappa;
    double fit_sigma_v;
    double fit_rho;
    double fit_lambda;
} FidelityMetrics;

// --- Sort Helpers ---
void sort_doubles_fast(double* arr, int n);

// --- Core ---
void compute_detrended_returns(double* ohlcv, int n_rows, double* out_returns);
double get_avg_volume(double* ohlcv, int n);

// --- Advanced Physics ---
int calc_hurst_horizon(double* returns, int max_len);
int detect_variance_break(double* returns, int n);

// --- Optimization ---
void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p);
void optimize_svcj_vol_weighted(double* ohlcv, int n, double dt, double avg_vol, SVCJParams* p, double* out_spot_vol);

// --- Engines ---
void run_full_audit_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out);
void run_instant_filter_vol(double ret, double vol, double avg_vol, double dt, SVCJParams* p, double* state, InstantState* out);

#endif