#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define SWARM_SIZE 5000
#define SIM_SCENARIOS 50
#define PATHS_PER_SCENARIO 200 // Higher paths for distribution fidelity
#define N_COLS 5

// --- Structures ---

typedef struct {
    double kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
    double v; 
    double weight;
} Particle;

typedef struct {
    double spot_vol_mean, innovation_z, ess, entropy;
} FilterStats;

typedef struct {
    double spread_bps;      // e.g. 0.0002 (2 bps)
    double impact_coef;     // Impact per unit vol
    double stop_sigma;
    double target_sigma_init;
    double decay_rate;      // Target decay per bar
    int horizon;
} MarketMicrostructure;

typedef struct {
    // Net Alpha (EV - Passive)
    double alpha_long;
    double alpha_short;
    
    // Statistical Discrimination
    double t_stat_long;
    double t_stat_short;
    
    // Separation Metric
    double cohens_d;        // Distance between Long/Short Distributions
    double overlap_prob;    // Probability that Long == Short (Confusion)
    
    // Integrity
    double friction_cost_avg;
} ContrastiveResult;

// --- Functions ---
void compute_log_returns(double* ohlcv, int n, double* out);
void generate_prior_swarm(double* ohlcv, int n, double dt, Particle* out);
void run_particle_filter_step(Particle* sw, double ret, double dt, FilterStats* out);

// The Contrastive Engine
void run_contrastive_simulation(
    Particle* swarm, 
    double price, 
    double dt, 
    MarketMicrostructure micro, 
    ContrastiveResult* out
);

#endif