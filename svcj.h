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
    double energy_ratio;
    double residue_bias;
    double hurst_exponent;
    
    double ks_stat;
    double levene_p;
    double jb_p;
    
    int is_valid;
} FidelityMetrics;

// Core
void compute_volume_weighted_returns(double* ohlcv, int n, double* out_ret);

// Optimization
void estimate_initial_params(double* ret, int n, double dt, SVCJParams* p);
double ukf_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot);
void optimize_svcj(double* returns, int n, double dt, SVCJParams* p, double* out_spot);

// Pipeline
void run_fidelity_scan_advanced(double* ohlcv, int total_len, int w_grav, int w_imp, double dt, FidelityMetrics* out);

// Helpers (Exposed for Wrapper if needed, or kept internal)
void sort_doubles_fast(double* arr, int n);

#endif