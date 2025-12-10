#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 300
#define N_COLS 5

// --- Structures ---

typedef struct {
    double mu;          // Structural Drift (Trend)
    double theta;       // Structural Variance (Noise Envelope)
    double kappa;       // Mean Reversion Speed
    double sigma_v;     // Vol of Vol (Stability)
    double rho;         // Correlation
    double lambda_j;    // Jump Intensity
    double mu_j;
    double sigma_j;
} SVCJParams;

typedef struct {
    int natural_window; // The detected stable window
    double min_sigma_v; // The stability metric
} FrequencyResult;

typedef struct {
    double max_deviation; // The Z-Score of the max excursion
    double p_value;       // Significance (Reflection Principle)
    double drift_term;    // The trend component subtracted
    double vol_term;      // The noise component normalized
    int is_breakout;      // 1 = True Structural Break
    int break_index;      // Which bar caused the break
} CausalStats;

// --- Prototypes ---

// 1. Natural Frequency Engine
void run_vov_spectrum_scan(double* ohlcv, int total_len, double dt, int step, FrequencyResult* out);

// 2. Physics Fitting
void fit_gravity_physics(double* ohlcv, int n, double dt, SVCJParams* out_params);

// 3. Causal Cone Test
void test_causal_cone(double* impulse_prices, int n_impulse, double dt, SVCJParams* gravity, CausalStats* out);

// Utils
void compute_log_returns(double* ohlcv, int n, double* out);

#endif