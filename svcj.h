#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define SWARM_SIZE 5000
#define SIM_SCENARIOS 50 // Representative particles to simulate
#define PATHS_PER_SCENARIO 100 // Paths per particle
#define N_COLS 5

// --- Structures ---

typedef struct {
    double kappa, theta, sigma_v, rho, lambda_j;
    double v; 
    double weight;
} Particle;

typedef struct {
    double spot_vol_mean, spot_vol_std, innovation_z;
    double entropy, ess;
} FilterStats;

// The Strategy Definition
typedef struct {
    int direction;          // 1 (Long), -1 (Short)
    double stop_sigma;      // Stop Loss width in Sigmas
    double target_sigma;    // Profit Target width in Sigmas
    int horizon_bars;       // Max holding time
} ActionProfile;

// The Valuation
typedef struct {
    double ev;              // Expected Value (Mean PnL)
    double std_dev;         // Dispersion of Outcomes
    double win_rate;        // % Paths Positive
    double t_stat;          // EV / (StdDev / sqrt(N)) -> The Discriminator
    double kelly_q;         // Suggested sizing (Kelly Criterion)
} SimulationResult;

// --- Functions ---
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);
void generate_prior_swarm(double* ohlcv, int n, double dt, Particle* out_swarm);
void run_particle_filter_step(Particle* current_swarm, double return_val, double dt, FilterStats* out_stats);

// The New Simulation Engine
void run_ev_simulation(Particle* swarm, double current_price, double dt, ActionProfile action, SimulationResult* out);

#endif