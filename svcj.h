#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define N_COLS 5
#define TICK_BUFFER_SIZE 60 // Buffer for Skew/Jerk stats
#define NM_ITER 250

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

// The High-Frequency State (Internal)
typedef struct {
    double price_buffer[TICK_BUFFER_SIZE];
    double z_score_buffer[TICK_BUFFER_SIZE];
    int buffer_idx;
    int is_full;
    double state_variance; // The UKF state
} TickState;

// The Instantaneous Output
typedef struct {
    double z_score;
    double jerk;
    double skew;
    int needs_recalibration; // Model Failure Flag
} InstantMetrics;

// --- Physics I/O ---
int save_physics(const char* ticker, SVCJParams* p);
int load_physics(const char* ticker, SVCJParams* p);

// --- Core ---
double ukf_likelihood(double* ret, double* vol, int n, double dt, SVCJParams* p);
void optimize_svcj_volume(double* ohlcv, int n, double dt, SVCJParams* p);

// --- The Instantaneous Engine ---
void init_tick_state(TickState* state, SVCJParams* p);
void run_tick_update(double price, double vol, double dt, SVCJParams* p, TickState* state, InstantMetrics* out);
void check_model_coherence(TickState* state, InstantMetrics* out); // KS Test

#endif