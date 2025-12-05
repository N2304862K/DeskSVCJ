#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Configuration
#define NM_ITER 300 // Increased for convergence without priors
#define SQRT_2PI 2.50662827463
#define N_COLS 5

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    double delta, gamma, vega, theta_decay;
} SVCJGreeks;

typedef struct {
    double ll_null;      // Long Window (Null Hypothesis)
    double ll_alt;       // Short Window (Alternative)
    double statistic;    // D = 2 * (LL_alt - LL_null)
    double p_value;
    int significant;
} RegimeStats;

// Core
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

// Optimization
void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p);
double ukf_log_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);

// Statistical Test
void perform_likelihood_test(double* ohlcv, int len_long, int len_short, double dt, RegimeStats* out);

// Pricing
void calc_greeks(double s0, double K, double T, double r, SVCJParams* p, double spot_vol, int type, SVCJGreeks* out);

#endif