#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define N_COLS 5
#define TICK_BUFFER_SIZE 30 // For Skew/Jerk

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    double z_score;
    double jerk; // d(Z)/dt
    double skew; // Skew of recent Z-scores
    int recalibrate_flag; // 1 = Model Failure
} InstantMetrics;

// --- GLOBAL STATE (Internal to C-Core) ---
// This allows the C-Core to be stateful across calls

// --- FUNCTION PROTOTYPES ---

// File I/O
int load_physics(const char* ticker, SVCJParams* p);
void save_physics(const char* ticker, SVCJParams* p);

// Core
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns, double* out_volumes);
void optimize_svcj_core(double* returns, double* volumes, int n, double dt, SVCJParams* p);

// The New Instantaneous Engine
void initialize_tick_engine(SVCJParams* p, double dt);
void run_tick_update(double price, double volume, InstantMetrics* out);

// The Calibration Engine (Called only when flagged)
void run_full_calibration(const char* ticker, double* ohlcv, int n, double dt);

#endif