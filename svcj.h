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
    double ll_constrained, ll_unconstrained, test_statistic, p_value;
    int is_significant;
} RegimeTestStats;

void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);
void estimate_initial_params_ohlcv(double* ohlcv, int n, double dt, SVCJParams* p);
double ukf_log_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob, double theta_proxy);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);
void perform_likelihood_ratio_test(double* ohlcv_long, int len_long, int len_short, double dt, RegimeTestStats* out);

#endif