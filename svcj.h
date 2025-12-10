#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define N_COLS 5 // OHLCV

// --- Structures ---

// A single hypothesis about the market's physics
typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j;
    double weight; // Fitness score
} Particle;

// A probabilistic belief (Gaussian) about the market's structure
typedef struct {
    double mean[6]; // Mean vector for the 6 core params
    double cov[36]; // 6x6 Covariance Matrix
} GravityDistribution;

// The output of the live filter
typedef struct {
    double expected_return;
    double expected_vol;
    double mahalanobis_dist; // Escape Velocity from Gravity
    double kl_divergence;    // Surprise Index
    double swarm_entropy;    // Swarm Health
} InstantState;

// --- Function Prototypes ---

// Utils
void compute_log_returns(double* ohlcv, int n, double* out_ret, double* out_vol);

// Gravity Engine (Low Frequency)
void run_gravity_scan(double* ohlcv, int total_len, double dt, GravityDistribution* out_anchor);

// Particle Filter (High Frequency)
void generate_prior_swarm(GravityDistribution* anchor, int n_particles, Particle* out_swarm);
void run_particle_filter_step(Particle* current_swarm, int n_particles, double new_return, double new_volume, double avg_vol, double dt, GravityDistribution* anchor, Particle* next_swarm, InstantState* out_state);

#endif