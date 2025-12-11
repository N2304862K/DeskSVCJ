#ifndef SVCJ_SWARM_H
#define SVCJ_SWARM_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define N_PARTICLES 2000
#define MIN_ particles 100
#define SQRT_2PI 2.50662827463

// The Hypothesis Unit
typedef struct {
    double v;           // Spot Variance
    double mu;          // Instantaneous Drift (Trend)
    double rho;         // Stochastic Correlation
    int jump_state;     // 1 = Jump Regime, 0 = Diffusion
    double weight;      // Probability of this hypothesis
} Particle;

// Hyperparameters (The Rules of the Game)
typedef struct {
    double kappa;
    double theta;       // Long Run Variance
    double sigma_v;     // Vol of Vol
    double lambda_j;    // Jump Intensity
    double mu_j;        // Jump Mean
    double sigma_j;     // Jump Std
    double rho_mean;    // Mean Reversion for Correlation
} SwarmParams;

// The Decision Payload
typedef struct {
    double expected_return;  // Weighted Mean Drift (EV Direction)
    double risk_volatility;  // 95th Percentile Vol (EV Cost)
    double swarm_entropy;    // Confusion Metric (Action Blurring)
    double regime_prob;      // Probability of Jump/Trend Regime
    double effective_rho;    // Consensus Correlation
} SwarmMetrics;

// Core Functions
void init_swarm(Particle* swarm, SwarmParams* p);
void predict_step(Particle* swarm, SwarmParams* p, double dt, double diurnal_factor);
void update_step(Particle* swarm, SwarmParams* p, double ret, double range_sq, double dt);
void resample_regularized(Particle* swarm, SwarmParams* p);
void calc_swarm_metrics(Particle* swarm, SwarmMetrics* out);

// Helpers
double get_random_normal();
double get_random_uniform();

#endif