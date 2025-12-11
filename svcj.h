#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

// 3 Swarms x 1000 Particles
#define N_SUB_PARTICLES 1000
#define N_TOTAL (N_SUB_PARTICLES * 3)

typedef struct {
    double v;
    double mu;
    double rho;
    double weight;
    double last_log_p;
} Particle;

typedef struct {
    // Regime Probabilities (The Signal)
    double prob_bull;
    double prob_bear;
    double prob_neutral;
    
    // Physics State (Weighted Average)
    double agg_vol;
    double agg_drift;
    double entropy;
} IMMState;

typedef struct {
    double kappa;
    double theta;
    double sigma_v;
    double lambda_j;
    double mu_j;
    double sigma_j;
} PhysicsParams;

void init_imm(PhysicsParams* phys, Particle* particles, double start_price);
void update_imm(Particle* particles, PhysicsParams* phys, 
                double o, double h, double l, double c, 
                double vol_ratio, double diurnal_factor, double dt, 
                IMMState* out);

#endif