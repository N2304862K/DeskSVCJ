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
    double current_spot_vol;
    double current_jump_prob;
    double innovation_z_score;
} InstantState;

typedef struct {
    int win_impulse, win_gravity;
    double energy_ratio;
    double levene_p, mw_p, ks_p;
    int is_valid;
    double residue_median;
    double fit_theta, fit_kappa, fit_sigma_v, fit_rho, fit_lambda;
} FidelityMetrics;

// --- Specialized Sort Struct ---
typedef struct {
    double val;
    int group;
    double rank;
} RankItem;

// Utils
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

// High-Performance Sorting (New)
void sort_doubles_fast(double* arr, int n);
void sort_ranks_fast(RankItem* arr, int n);

// Optimization
void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p);
void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob);

// Engines
void run_nonparametric_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out);
void run_instant_filter(double return_val, double dt, SVCJParams* p, double* state_var, InstantState* out);

#endif