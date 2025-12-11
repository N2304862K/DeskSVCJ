#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define SWARM_SIZE 5000
#define SIM_SCENARIOS 50
#define PATHS_PER_SCENARIO 200
#define N_COLS 5

typedef struct {
    double kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
    double v; 
    double weight;
} Particle;

typedef struct {
    double spot_vol_mean, innovation_z, ess, entropy;
} FilterStats;

typedef struct {
    double spread_bps, impact_coef, stop_sigma, target_sigma_init, decay_rate;
    int horizon;
} MarketMicrostructure;

typedef struct {
    double alpha_long, alpha_short;
    double t_stat_long, t_stat_short;
    double cohens_d, friction_cost_avg;
} ContrastiveResult;

void compute_log_returns(double* ohlcv, int n, double* out);
void generate_prior_swarm(double* ohlcv, int n, double dt, Particle* out);
void run_particle_filter_step(Particle* sw, double ret, double dt, FilterStats* out);
void run_contrastive_simulation(Particle* sw, double p, double dt, MarketMicrostructure m, ContrastiveResult* r);

#endif