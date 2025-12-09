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
    // Physics
    double energy_ratio;    // Model Kinetic / Model Potential
    double residue_bias;    // Direction
    double hurst_exponent;  // Persistence
    
    // Statistics
    double realized_f_stat; // Realized Var Impulse / Realized Var Gravity
    double f_p_value;       // Significance of Expansion
    double ks_stat;         // Distribution Shift
    double jb_p;            // Tail Risk Significance
    
    int is_valid;
} FidelityMetrics;

void compute_volume_weighted_returns(double* ohlcv, int n, double* out_ret);
void optimize_svcj_volume(double* returns, double* volumes, int n, double dt, SVCJParams* p, double* out_spot);
void run_fidelity_scan_advanced(double* ohlcv, int total_len, int w_grav, int w_imp, double dt, FidelityMetrics* out);

#endif