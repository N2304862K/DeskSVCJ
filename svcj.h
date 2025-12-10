#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NM_ITER 200
#define N_COLS 5
#define MAX_PARTICLES 2000

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
    double weight;
} Particle;

typedef struct {
    double mean[8];
    double cov[64];
} GravityDistribution;

typedef struct {
    double expected_return;
    double expected_vol;
    double escape_velocity; // Mahalanobis Dist
    double surprise_index;  // KL Divergence
    double swarm_entropy;
} InstantMetrics;

// This struct holds the entire state of the evolving system
typedef struct {
    GravityDistribution anchor;
    Particle swarm[MAX_PARTICLES];
    int n_particles;
    double dt;
    double avg_volume;
} EvolvingSystemState;

// --- Function Prototypes ---
void compute_log_returns(double* ohlcv, int n, double* out_ret, double* out_vol);

// Initialization
void initialize_system(double* ohlcv, int n, double dt, int n_particles, EvolvingSystemState* out_state);

// The Main Loop
void run_system_step(EvolvingSystemState* state, double new_return, double new_volume, InstantMetrics* out_metrics);

// Internal Helpers (Not exposed to Cython)
void optimize_single_window(double* ret, double* vol, int n, double dt, SVCJParams* p);
void check_constraints(SVCJParams* p);

#endif