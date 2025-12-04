#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Tuning Constants
#define NM_ITER 200
#define SQRT_2PI 2.50662827463
#define N_COLS 5 // OHLCV Columns

// Parameter Structure
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

// Greeks Structure
typedef struct {
    double delta;
    double gamma;
    double vega;
    double theta_decay;
} SVCJGreeks;

// Core Utils
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

// Optimization & Filtering
void estimate_initial_params_ohlcv(double* ohlcv, int n, double dt, SVCJParams* p);
double ukf_log_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob, double realized_theta_proxy);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);

// Pricing & Risk
void calc_svcj_greeks(double s0, double K, double T, double r, SVCJParams* p, double spot_vol, int type, SVCJGreeks* out);

#endif