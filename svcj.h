#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 200
#define SQRT_2PI 2.50662827463
#define N_COLS 5 // Open,High,Low,Close,Vol

// 1. Parameter State
typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

// 2. Statistical Regime Test (Wilks' Lambda)
typedef struct {
    double ll_constrained;   // Null Hypothesis (Long Memory)
    double ll_unconstrained; // Alt Hypothesis (Short Memory)
    double test_statistic;   // D value
    double p_value;          // Confidence
    int is_significant;      // Binary Flag
} RegimeTestStats;

// 3. Gradient / Surface Physics
typedef struct {
    double slope;            // Term Structure Slope (dTheta / dLnT)
    double curvature;        // Convexity
    double short_theta;
    double long_theta;
} GradientStats;

// Core Functions
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);
void estimate_initial_params_ohlcv(double* ohlcv, int n, double dt, SVCJParams* p);

// The Engines
double ukf_log_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob, double theta_proxy);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);

// Advanced Analysis
void perform_likelihood_ratio_test(double* ohlcv, int len_long, int len_short, double dt, RegimeTestStats* out);
void calculate_structural_gradient(double* ohlcv, int len, double dt, GradientStats* out);

#endif