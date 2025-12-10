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
    // --- PHYSICS PAYLOAD (Exposed to Python) ---
    // Core Test
    double max_deviation;
    double p_value;
    int is_breakout;
    
    // Confirmation Factors
    double hurst_exponent;
    double residue_bias;
    double energy_ratio;
    
} CausalStats;

// Core
void compute_log_returns(double* ohlcv, int n, double* out);

// Optimization
void optimize_svcj(double* ret, int n, double dt, SVCJParams* p);

// Main Engine
void test_causal_cone(double* impulse_prices, int n_impulse, double dt, SVCJParams* gravity, CausalStats* out);
void fit_gravity_physics(double* ohlcv, int n, double dt, SVCJParams* out_params);

// Helpers
double calc_hurst(double* data, int n);

#endif