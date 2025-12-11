#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define N_PARTICLES 2000
#define MIN_EFFECTIVE_PARTICLES 1000
#define CHI_SQ_CUTOFF 16.0 

// Dynamic Particle (Learns Physics)
typedef struct {
    double v;           // Variance
    double mu;          // Drift
    double theta;       // Dynamic Structural Vol
    double kappa;       // Dynamic Mean Reversion
    double rho;         // Correlation
    double weight;      // Probability
    double last_log_p;  // Memory
} Particle;

// Multi-Scale Momentum Memory
typedef struct {
    double trend_fast;  // ~5 min
    double trend_mid;   // ~15 min
    double trend_slow;  // ~60 min
    double coherence;   // Alignment score (-1 to 1)
} TrendState;

// Output
typedef struct {
    double ev_vol;
    double mode_vol;
    double ev_drift;
    double entropy;
    int collapsed;
    
    // Debugging / Telematics
    double global_trend_coherence;
    double avg_theta; // Swarm's view of gravity
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
                  SwarmState* out);

#endif