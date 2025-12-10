#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define N_COLS 5
#define SWARM_SIZE 5000 // Number of particles

// --- Structures ---

// Physics of a single "Hypothesis" (Particle)
typedef struct {
    double kappa;
    double theta;
    double sigma_v;
    double rho;
    double lambda_j;
    
    // Hidden State
    double v; // Current Variance State
    
    // Metadata
    double weight;
} Particle;

// System Health & Output
typedef struct {
    // Outputs
    double spot_vol_mean;
    double spot_vol_std;
    double innovation_z;
    
    // Health Checks
    double entropy;
    double ess; // Effective Sample Size
} FilterStats;

// --- Function Prototypes ---

// Core Utils
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);
double normal_rng(double mu, double std);
double lognormal_rng(double mu, double std);

// Pipeline
void generate_prior_swarm(double* ohlcv, int n, double dt, Particle* out_swarm);
void run_particle_filter_step(Particle* current_swarm, double return_val, double dt, FilterStats* out_stats);

#endif