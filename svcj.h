#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 400
#define SQRT_2PI 2.50662827463
#define N_COLS 5

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    double se_theta;    // Standard Error of Theta
    double se_kappa;
    double se_sigma_v;
} HessianMetrics;

typedef struct {
    // Windows
    int win_gravity;
    int win_impulse;
    
    // Physics
    double theta_gravity;
    double theta_impulse;
    double energy_ratio;
    
    // Confidence
    double theta_std_err; // From Hessian
    double param_z_score; // |Theta_diff| / SE
    
    // Non-Parametric Stats
    double ad_stat;       // Anderson-Darling
    double hurst;
    double residue_bias;
    
    // Decision
    int is_valid;
} FidelityMetrics;

typedef struct {
    int window;
    double sigma_v;
} VoVPoint;

// Core
void compute_log_returns(double* ohlcv, int n, double* out_ret);
void compute_volume_weighted_returns(double* ohlcv, int n, double* out_ret);

// Optimization & Hessian
double ukf_likelihood(double* ret, int n, double dt, SVCJParams* p);
void optimize_svcj(double* ret, int n, double dt, SVCJParams* p);
void calculate_hessian_errors(double* ret, int n, double dt, SVCJParams* p, HessianMetrics* out);

// Engines
void run_vov_scan(double* ohlcv, int total_len, double dt, int step, VoVPoint* out_buf, int max_steps);
void run_fidelity_pipeline(double* ohlcv, int total_len, double dt, FidelityMetrics* out);

#endif