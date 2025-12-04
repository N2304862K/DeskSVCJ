#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 200
#define SQRT_2PI 2.50662827463
#define N_COLS 5 // Open,High,Low,Close,Vol

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    double ll_ratio_stat;    // -2 * ln(L_null / L_alt)
    double p_value;          // Chi-Square P-Value
    double divergence;       // (Theta_Short - Theta_Long) / Theta_Long
    double short_theta;
    double long_theta;
} RegimeMetrics;

void compute_log_returns(double* ohlcv, int n, double* out);
void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p);
double ukf_log_likelihood(double* ret, int n, double dt, SVCJParams* p, double* sv, double* jp, double proxy);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p);

// New: Full Pipeline Logic in C for speed
void run_structural_test(double* ohlcv, int len_long, int len_short, double dt, RegimeMetrics* out);

#endif