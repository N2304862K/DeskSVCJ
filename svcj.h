#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Configuration
#define DT (1.0/252.0)
#define NM_ITER 200
#define SQRT_2PI 2.50662827463

// Data Column Indices (OHLCV)
#define IDX_OPEN 0
#define IDX_HIGH 1
#define IDX_LOW 2
#define IDX_CLOSE 3
#define IDX_VOL 4
#define N_COLS 5

typedef struct {
    double mu;
    double kappa;
    double theta;
    double sigma_v;
    double rho;
    double lambda_j;
    double mu_j;
    double sigma_j;
} SVCJParams;

// Core Functions
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

// Optimization & Filtering
void estimate_initial_params_ohlcv(double* ohlcv, int n, SVCJParams* p);
double ukf_log_likelihood(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob, double realized_theta_proxy);
void optimize_svcj(double* ohlcv, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);

// Pricing
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double spot_vol, double* out_prices);

#endif