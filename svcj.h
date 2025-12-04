#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 200
#define SQRT_2PI 2.50662827463
#define N_COLS 5 // OHLCV

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    double delta, gamma, vega, theta_decay;
} SVCJGreeks;

// Core Functions
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

// Optimization & Filtering (Now accepts dt)
void estimate_initial_params_ohlcv(double* ohlcv, int n, double dt, SVCJParams* p);
double ukf_log_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob, double realized_theta);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);

// Pricing & Risk
void calc_svcj_greeks(double s0, double K, double T, double r, SVCJParams* p, double spot_vol, int type, SVCJGreeks* out_greeks);

#endif