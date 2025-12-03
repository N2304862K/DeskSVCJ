#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Configuration
#define DT (1.0/252.0)
#define NM_ITER_INIT 400   // Deep search for first window
#define NM_ITER_UPDATE 80  // Fast update for rolling windows
#define SQRT_2PI 2.50662827463
#define IDX_OPEN 0
#define IDX_HIGH 1
#define IDX_LOW 2
#define IDX_CLOSE 3
#define IDX_VOL 4
#define N_COLS 5

typedef struct {
    double mu;
    double kappa;
    double theta;
    double sigma_v;
    double rho;
    double lambda_j;
    double mu_j;
    double sigma_j;
} SVCJParams;

// Core Utils
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);
void clean_returns(double* returns, int n);
void check_constraints(SVCJParams* params);

// Optimization
double ukf_log_likelihood(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob, double theta_anchor);
void estimate_initial_params_smart(double* ohlcv, int n, SVCJParams* p);
void grid_search_init(double* returns, int n, SVCJParams* p, double theta_anchor);

// The New Fast Rolling Function
void optimize_rolling_chain(double* ohlcv_full, int n_total_days, int window, double* out_results);

// Pricing (Kept for completeness)
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double spot_vol, double* out_prices);

#endif