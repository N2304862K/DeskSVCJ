#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define N_PARTICLES 3000
#define N_COLS 5 // OHLCV
#define SQRT_2PI 2.50662827463

// The State of a Single Particle
typedef struct {
    double v;            // Current Variance
    double mu;           // Current Drift (Mean-Reverting)
    double weight;       // The Likelihood
    int regime;          // 0=Bull, 1=Bear, 2=Neutral
} Particle;

// The Host Filter State
typedef struct {
    // Aggregates
    double p_bull;
    double p_bear;
    double p_neutral;
    double entropy;
    
    // Core SVCJ Physics (Fixed per filter instance)
    double kappa;
    double theta;
    double sigma_v;
    double dt;
} IMMState;

// Core functions exposed to Cython
void init_particle_set(Particle* particles, double theta, double start_price);
void update_particles(Particle* particles, IMMState* state, double o, double h, double l, double c, double vol_factor, double range_factor);

#endif