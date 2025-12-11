#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define N_PARTICLES 2000
#define MIN_EFFECTIVE_PARTICLES 1000

// The Particle (Hypothesis)
typedef struct {
    double v;           // Variance
    double mu;          // Drift (Trend)
    double rho;         // Correlation
    double weight;      // Likelihood
    double last_log_p;  // Memory
} Particle;

// The Consensus (Output)
typedef struct {
    double ev_vol;      // Mean Vol (Risk)
    double mode_vol;    // Robust Vol (Signal)
    double ev_drift;    // Expected Trend
    double entropy;     // Confidence
} SwarmState;

// Physics Limits
typedef struct {
    double kappa;
    double theta;
    double sigma_v;
    double lambda_j;
    double mu_j;
    double sigma_j;
} PhysicsParams;

void init_swarm(PhysicsParams* phys, Particle* swarm, double start_price);
void update_swarm(Particle* swarm, PhysicsParams* phys, 
                  double o, double h, double l, double c, 
                  double vol_ratio, double diurnal_factor, double dt, 
                  SwarmState* out);

#endif