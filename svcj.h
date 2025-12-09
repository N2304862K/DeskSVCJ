#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

// Constants
#define NM_ITER 250
#define N_COLS 5 

// 1. Physics Parameters
typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

// 2. Instantaneous State (Monitoring)
typedef struct {
    double spot_vol;
    double z_score;
    double jump_prob;
    double sprt_score;
} InstantState;

// 3. Fidelity Metrics (Scanning)
typedef struct {
    int win_impulse, win_gravity, is_valid;
    double energy_ratio, residue_bias;
    double ks_stat, hurst_exp;
    // Fitted Physics to return to Python
    double fit_theta, fit_kappa, fit_sigma_v, fit_rho, fit_lambda;
} FidelityMetrics;

// 4. VoV Spectrum Point
typedef struct {
    int window;
    double sigma_v;
    double theta;
} VoVPoint;

// --- Function Prototypes ---

// Core Utils
void detrend_log_returns(double* ohlcv, int n, double* out_ret, double* out_vol_scale);

// The Core Engines
void run_fidelity_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out);
void run_instant_filter(double val, double vol, double avg_vol, double dt, SVCJParams* p, double* state_var, double* sprt_accum, InstantState* out);
void run_vov_scan(double* ohlcv, int total_len, double dt, int step, VoVPoint* out_buffer, int max_steps);

#endif