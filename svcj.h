#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 300
#define N_COLS 5

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    double energy_ratio;    // Theta_Impulse / Theta_Gravity
    double residue_bias;
    double hurst_exponent;  // On Raw Returns (valid)
    
    double ks_stat;
    double levene_p;
    double jb_p;
    
    int is_valid;
} FidelityMetrics;

// Core
void compute_log_returns(double* ohlcv, int n, double* out_ret, double* out_vol);

// Optimization
double ukf_volume_likelihood(double* ret, double* vol, int n, double dt, double avg_vol, SVCJParams* p);
void optimize_svcj(double* ret, double* vol, int n, double dt, SVCJParams* p);

// Pipeline
void run_fidelity_scan_native(double* ohlcv, int total_len, int w_grav, int w_imp, double dt, FidelityMetrics* out);

// Helpers
void sort_doubles_fast(double* arr, int n);

#endif