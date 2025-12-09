#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 300
#define N_COLS 5 // Open, High, Low, Close, Volume

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    // Physical Metrics
    int win_impulse;
    int win_gravity;
    double energy_ratio;
    double hurst_exponent; // Improvement 2
    
    // Robust Stats (Non-Parametric)
    double levene_p;       // Energy Validity
    double mw_p;           // Direction Validity (Raw)
    double ks_p_vol;       // Dist Match: Spot Vol (Impulse) vs Spot Vol (Gravity) (Imp 4)
    
    // Result
    int is_valid;
    double residue_median; // From Raw Returns
    
    // Physics Payload
    double fit_theta;
} FidelityMetrics;

// --- Sort Utils ---
void sort_doubles_fast(double* arr, int n);

// --- Core ---
// Improvement 5: Detrending logic
void prepare_data(double* ohlcv, int n, double* out_raw_ret, double* out_detrend_ret, double* out_vol_weights);

// Optimization (Volume Aware)
void optimize_svcj_vol(double* returns, double* vol_weights, int n, double base_dt, SVCJParams* p, double* out_spot);

// Engines
void run_enhanced_scan(double* ohlcv, int total_len, double base_dt, FidelityMetrics* out);

#endif