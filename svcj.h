#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 300
#define SQRT_2PI 2.50662827463
#define N_COLS 5

// --- Structures ---

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    // Physical Metrics
    double energy_ratio;
    double residue_bias;
    double hurst_exponent;
    double snr_ratio;
    
    // Statistical Tests (P-Values or Stats)
    double ks_stat;       // Kolmogorov-Smirnov
    double mwu_stat;      // Mann-Whitney U
    double levene_p;      // Variance Homogeneity
    double jb_p;          // Normality
    double ad_stat;       // Anderson-Darling (Tails)
    
    // Integrity
    int is_valid;         // Master Switch
    double vol_clock_factor; // Average Volume Multiplier
} FidelityMetrics;

// --- Function Prototypes ---

// Helpers
void sort_doubles_fast(double* arr, int n);
void sort_ranks_fast(double* arr, int* indices, int n);

// Core Physics
void compute_detrended_returns(double* ohlcv, int n, double* out_ret, double* out_vol);
void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p);
double ukf_volume_likelihood(double* returns, double* volumes, int n, double dt, double avg_vol, SVCJParams* p, double* out_spot);
void optimize_svcj_volume(double* returns, double* volumes, int n, double dt, SVCJParams* p, double* out_spot);

// Statistical Engines
void perform_ks_test(double* d1, int n1, double* d2, int n2, double* out_stat);
void perform_mann_whitney_u(double* d1, int n1, double* d2, int n2, double* out_stat);
void perform_levene_test(double* d1, int n1, double* d2, int n2, double* out_p);
void perform_jarque_bera(double* data, int n, double* out_p);
double calc_hurst_exponent(double* data, int n);

// Main Pipeline
void run_fidelity_scan_advanced(double* ohlcv, int total_len, int w_grav, int w_imp, double dt, FidelityMetrics* out);

#endif