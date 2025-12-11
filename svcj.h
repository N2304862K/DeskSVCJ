#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define N_PARTICLES 2000
#define MIN_EFFECTIVE_PARTICLES 1000

// Lean Particle (State Only)
typedef struct {
    double v;
    double mu;
    double rho;
    double weight;
    double last_log_p;
} Particle;

typedef struct {
    double trend_fast;
    double trend_mid;
    double trend_slow;
    double coherence;
} TrendState;

typedef struct {
    double ev_vol;
    double mode_vol;
    double ev_drift;
    double entropy;
    int collapsed;
    double trend_coherence;
} SwarmState;

typedef struct {
    double kappa;
    double theta;
    double sigma_v;
    double lambda_j;
    double mu_j;
    double sigma_j;
} PhysicsParams;

void init_swarm(PhysicsParams* phys, Particle* swarm, double start_price, TrendState* ts);
void update_swarm(Particle* swarm, TrendState* ts, PhysicsParams* phys, 
                  double o, double h, double l, double c, 
                  double vol_ratio, double diurnal_factor, double dt, 
                  double prev_entropy,
                  SwarmState* out);

#endif