#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Configuration
#define MULTI_START_POINTS 5  // Number of random restarts to avoid local minima
#define NM_ITER 300           // Deep iteration for accuracy
#define SQRT_2PI 2.50662827463
#define N_COLS 5

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
    double final_likelihood; // Store result quality
} SVCJParams;

typedef struct {
    double ll_ratio;
    double p_value;
    double short_theta;
    double long_theta;
    double divergence;
} SnapshotStats;

void compute_log_returns(double* ohlcv, int n, double* out);

// Core Logic
double raw_log_likelihood(double* returns, int n, double dt, SVCJParams* p);
void optimize_snapshot_raw(double* ohlcv, int n, double dt, SVCJParams* p);

// Pipeline
void run_snapshot_test(double* ohlcv, int w_long, int w_short, double dt, SnapshotStats* out);

#endif