#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Configuration
#define NM_ITER 300
#define SQRT_2PI 2.50662827463
#define N_COLS 5 // Open, High, Low, Close, Volume

// 1. The Physics (Parameters)
typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

// 2. The State (Instantaneous)
typedef struct {
    double current_spot_vol;
    double current_jump_prob;
    double innovation_z_score;
} InstantState;

// 3. The Spectrum Point (For VoV Scan)
typedef struct {
    int window;
    double sigma_v; // Stability Metric
    double theta;   // Structural Vol
} VoVPoint;

// 4. The Fidelity Report (Statistical Validation)
typedef struct {
    // Physicals
    int win_impulse;
    int win_gravity;
    double energy_ratio;
    double residue_bias;
    
    // Stats
    double f_p_value;    // Variance Ratio Significance
    double t_p_value;    // Drift Significance
    double lb_p_value;   // Momentum Significance (Ljung-Box)
    int is_valid;        // 1 = Pass, 0 = Fail
    
    // Physics Export (To sync Monitor)
    double fit_theta;
    double fit_kappa;
    double fit_sigma_v;
    double fit_rho;
    double fit_lambda;
} FidelityMetrics;

// 5. Greeks
typedef struct {
    double delta, gamma, vega, theta_decay;
} SVCJGreeks;

// --- Function Prototypes ---

// Utils
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

// Optimization Core
void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p);
double ukf_pure_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);

// Engines
void run_vov_scan(double* ohlcv, int total_len, double dt, int step, VoVPoint* out_buffer, int max_steps);
void run_fidelity_check(double* ohlcv, int total_len, int win_grav, int win_imp, double dt, FidelityMetrics* out);
void run_instant_filter(double return_val, double dt, SVCJParams* p, double* state_var, InstantState* out);

// Pricing
void calc_greeks(double s0, double K, double T, double r, SVCJParams* p, double spot_vol, int type, SVCJGreeks* out);

#endif