#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define DT_DAILY (1.0/252.0)
#define NM_ITER 300
#define SQRT_2PI 2.50662827463
#define N_COLS 5

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    double delta, gamma, vega, theta_decay;
} SVCJGreeks;

void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);
void estimate_initial_params_ohlcv(double* ohlcv, int n, double dt, SVCJParams* p);
double ukf_log_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob, double realized_theta_proxy);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);
void calc_svcj_greeks(double s0, double K, double T, double r, SVCJParams* p, double spot_vol, int type, SVCJGreeks* out);

#endif