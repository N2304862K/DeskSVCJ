#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define N_PARTICLES 2000
#define MIN_EFFECTIVE_PARTICLES 1000
#define DT_BASE (1.0/(252.0*78.0)) // 5-minute bars default

// The Hypothesis (Particle)
typedef struct {
    double v;           // Spot Variance
    double mu;          // Instantaneous Drift (Trend)
    double rho;         // Correlation (Stochastic)
    double weight;      // Probability of this hypothesis
    double last_log_p;  // For likelihood calc
} Particle;

// The Consensus (Swarm Output)
typedef struct {
    double ev_vol;      // Expected Value (Mean) Volatility
    double mode_vol;    // Most Probable Volatility (Robust)
    double ev_drift;    // Expected Trend
    double entropy;     // Swarm Confusion (Action Blurring)
    double jump_prob;   // Probability of Jump State
    
    // Limits
    double vol_99;      // 99th Percentile Vol (Risk Limit)
} SwarmState;

// Static Physics (The "Rules")
typedef struct {
    double kappa;
    double theta;
    double sigma_v;
    double lambda_j;
    double mu_j;
    double sigma_j;
} PhysicsParams;

// Core Functions
void init_swarm(PhysicsParams* phys, Particle* swarm, double start_price);
void update_swarm(Particle* swarm, PhysicsParams* phys, 
                  double open, double high, double low, double close, 
                  double vol_ratio, double diurnal_factor, double dt, 
                  SwarmState* out_state);

#endif