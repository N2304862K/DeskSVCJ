#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define N_PARTICLES 2000
#define MIN_EFFECTIVE_PARTICLES 1000
#define CHI_SQ_CUTOFF 9.0

typedef struct {
    double v;
    double mu;
    double rho;
    double weight;
    double last_log_p;
} Particle;

typedef struct {
    double ev_vol;
    double mode_vol;
    double ev_drift;
    double entropy;
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
void update_swarm(Particle* swarm, PhysicsParams* phys, 
                  double o, double h, double l, double c, 
                  double vol_ratio, double diurnal_factor, double dt,
                  double context_drift, double context_conf, // NEW INPUTS
                  SwarmState* out);

#endif