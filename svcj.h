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
    double current_spot_vol;
    double current_jump_prob;
    double innovation_z_score;
    double kalman_variance; // P_k (State Uncertainty)
} InstantState;

typedef struct {
    // 1. Hessian (Parameter Uncertainty)
    double theta_std_err;
    double kappa_std_err;
    
    // 2. Normality (Jarque-Bera)
    double skewness;
    double kurtosis;
    double jb_stat;
    double jb_p_value;
    
    // 3. Consistency (Rolling Sigma_V)
    double realized_vol_of_vol;
    double model_vol_of_vol;
    double vov_ratio; // Realized / Model
    
    // 4. Resonance (Valley Width)
    double valley_sharpness; 
    
    // 5. Greeks Range (Sensitivity)
    double delta_lower;
    double delta_upper;
} ValidationReport;

typedef struct {
    double energy_ratio;
    double residue_bias;
    double f_p_value;
    double t_p_value;
    int is_valid;
    
    // Physics Export
    double fit_theta;
    double fit_kappa;
    double fit_sigma_v;
    double fit_rho;
    double fit_lambda;
} FidelityMetrics;

// Utils
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

// Optimization
void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p);
double ukf_pure_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob, double* out_kalman_var);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);

// Engines
void run_fidelity_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out);
void run_instant_filter(double return_val, double dt, SVCJParams* p, double* state_var, InstantState* out);

// NEW: The Validation Engine (8 Improvements)
void validate_fidelity(double* ohlcv, int n, double dt, SVCJParams* p, ValidationReport* out);

#endif