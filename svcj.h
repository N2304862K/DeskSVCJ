#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define N_PARTICLES 3000 // Increased for 3 sub-swarms
#define MIN_EFFECTIVE_PARTICLES 1500

typedef struct {
    double v;           // Variance
    double mu;          // Drift
    double rho;         // Correlation
    double weight;      // Probability
    double last_log_p;  // Memory
    int regime;         // 1=Bull, -1=Bear, 0=Neutral
} Particle;

typedef struct {
    double mode_vol;
    double entropy;
    
    // Regime Probabilities (The Clean Signal)
    double prob_bull;
    double prob_bear;
    double prob_neutral;
    
    int collapsed;
} SwarmState;

typedef struct {
    double kappa;
    double theta;
    double sigma_v;
    double lambda_j;
    double mu_j;
    double sigma_j;
} PhysicsParams;

void init_swarm(PhysicsParams* phys, Particle* swarm, double start_price);
void update_swarm_regime(Particle* swarm, PhysicsParams* phys, 
                         double o, double h, double l, double c, 
                         double vol_ratio, double diurnal_factor, double dt, 
                         SwarmState* out);

#endif