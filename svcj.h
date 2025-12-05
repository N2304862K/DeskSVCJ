#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 300
#define SQRT_2PI 2.50662827463
#define N_COLS 5

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    int window_len;
    double theta;        // Structural Vol
    double spot_vol;     // Kinetic Vol
    double residue_sum;  // Directional Bias
} SpectralPoint;

// Core
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

// Optimization
void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p);
double ukf_pure_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);

// New: Spectral Scan Engine
void run_spectral_scan(double* ohlcv, int total_len, int* windows, int n_windows, double dt, SpectralPoint* out_results);

#endif