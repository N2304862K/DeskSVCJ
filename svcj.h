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
    double energy_ratio;    // Kinetic (Spot) / Potential (Theta)
    double drift_impulse;   // Short-Term Drift - Long-Term Drift
    double jump_dominance;  // % of variance coming from Jumps
    double residue_bias;    // Sum of residuals (Directional Pressure)
    int is_breakout;        // Binary Classifier
} BreakoutStats;

// Core
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

// Optimization
void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p);
double ukf_pure_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);

// The Breakout Engine
void calculate_breakout_physics(double* ohlcv, int len_long, int len_short, double dt, BreakoutStats* out);

#endif