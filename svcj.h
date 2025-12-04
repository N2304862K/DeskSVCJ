#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 200
#define SQRT_2PI 2.50662827463
#define N_COLS 5

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    double delta, gamma, vega, theta_decay;
} SVCJGreeks;

// Statistical Test Results
typedef struct {
    double ll_constrained;   // Likelihood of Short Data given Long Params
    double ll_unconstrained; // Likelihood of Short Data given Optimized Params
    double test_statistic;   // Wilks' Lambda (Chi-Square)
    double p_value;          // Statistical Confidence of Break
    int is_significant;      // 1 if Break is real, 0 if Noise
} RegimeTestStats;

void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

void estimate_initial_params_ohlcv(double* ohlcv, int n, double dt, SVCJParams* p);
double ukf_log_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob, double realized_theta_proxy);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);

// New: Rigorous Regime Test
void perform_likelihood_ratio_test(double* ohlcv_long, int len_long, int len_short, double dt, RegimeTestStats* out_stats);

void calc_svcj_greeks(double s0, double K, double T, double r, SVCJParams* p, double spot_vol, int type, SVCJGreeks* out);

#endif