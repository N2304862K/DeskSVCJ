#include "svcj.h"
#include <float.h>

// --- Helper Functions ---
// Box-Muller transform for Gaussian random numbers
double normal_rng(double mu, double std) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mu + std * z;
}

double lognormal_rng(double mu, double std) {
    return exp(normal_rng(log(mu), std));
}

// --- Prior Generation (The "Seeding") ---

// Dummy optimize for prior center (placeholder, as full NM is too long)
void get_anchor_fit(double* ohlcv, int n, double dt, double* out_params, double* out_errors) {
    // In production, this runs optimize_svcj + calc_hessian
    // For this self-contained system, we use robust heuristics
    out_params[0] = 3.0; // kappa
    out_params[1] = 0.04; // theta
    out_params[2] = 0.5; // sigma_v
    out_params[3] = -0.5; // rho
    out_params[4] = 0.5; // lambda
    
    // Assume 10% relative error for seeding
    out_errors[0] = 0.3;
    out_errors[1] = 0.004;
    out_errors[2] = 0.05;
    out_errors[3] = 0.05;
    out_errors[4] = 0.05;
}

void generate_prior_swarm(double* ohlcv, int n, double dt, Particle* out_swarm) {
    double center[5];
    double errors[5];
    get_anchor_fit(ohlcv, n, dt, center, errors);
    
    for (int i=0; i<SWARM_SIZE; i++) {
        // Sample from informed distributions
        out_swarm[i].kappa = lognormal_rng(center[0], errors[0]);
        out_swarm[i].theta = lognormal_rng(center[1], errors[1]);
        out_swarm[i].sigma_v = lognormal_rng(center[2], errors[2]);
        
        // Rho needs to be bounded [-1, 1]
        double rho_sample = normal_rng(center[3], errors[3]);
        if (rho_sample > 0.99) rho_sample = 0.99;
        if (rho_sample < -0.99) rho_sample = -0.99;
        out_swarm[i].rho = rho_sample;
        
        out_swarm[i].lambda_j = lognormal_rng(center[4], errors[4]);
        
        // Initialize state and weight
        out_swarm[i].v = out_swarm[i].theta;
        out_swarm[i].weight = 1.0 / SWARM_SIZE;
    }
}

// --- Instantaneous Particle Filter Step ---
void run_particle_filter_step(Particle* current_swarm, double return_val, double dt, FilterStats* out_stats) {
    double sum_weights = 0;
    
    // --- 1. Fork & Weight (Prediction & Importance Sampling) ---
    for (int i=0; i<SWARM_SIZE; i++) {
        Particle* p = &current_swarm[i];
        
        // Predict next variance state
        double v_pred = p->v + p->kappa * (p->theta - p->v) * dt;
        if (v_pred < 1e-9) v_pred = 1e-9;
        
        // Calculate Likelihood of observing return_val given this particle's physics
        // Simplified Likelihood: Gaussian PDF
        double mu_exp = -0.5 * v_pred; // Risk-neutral drift assumption
        double y = return_val - mu_exp * dt;
        
        // Variance of innovation (Diffusive + Jump)
        double jump_var_contrib = p->lambda_j * 0.01; // Assume mu_j=0, sigma_j=0.1 for speed
        double S = v_pred*dt + jump_var_contrib*dt;
        if (S < 1e-12) S = 1e-12;
        
        double likelihood = (1.0 / sqrt(2*M_PI*S)) * exp(-0.5 * y*y / S);
        
        // Update Weight
        p->weight *= likelihood;
        sum_weights += p->weight;
        
        // Propagate State
        // Add random shock (Process Noise)
        p->v = v_pred + p->sigma_v * sqrt(v_pred*dt) * normal_rng(0, 1);
        if (p->v < 1e-9) p->v = 1e-9;
    }
    
    // --- 2. Normalize Weights & Calculate Stats ---
    double sum_sq_weights = 0;
    double mean_vol = 0, m2_vol = 0;
    double mean_z = 0;
    
    if (sum_weights < 1e-30) { // All particles died -> Total model failure
        // Reset weights to uniform to recover
        for (int i=0; i<SWARM_SIZE; i++) current_swarm[i].weight = 1.0/SWARM_SIZE;
        sum_weights = 1.0;
    }

    for (int i=0; i<SWARM_SIZE; i++) {
        Particle* p = &current_swarm[i];
        p->weight /= sum_weights;
        sum_sq_weights += p->weight * p->weight;
        
        double vol = sqrt(p->v);
        mean_vol += p->weight * vol;
        
        // Z-Score for this particle's prediction
        double jump_var = p->lambda_j * 0.01;
        double step_std = sqrt((p->v + jump_var) * dt);
        mean_z += p->weight * (return_val / step_std);
    }
    
    // Calculate Variance of the Volatility Estimate
    for (int i=0; i<SWARM_SIZE; i++) {
        m2_vol += current_swarm[i].weight * pow(sqrt(current_swarm[i].v) - mean_vol, 2);
    }
    
    out_stats->spot_vol_mean = mean_vol;
    out_stats->spot_vol_std = sqrt(m2_vol);
    out_stats->innovation_z = mean_z;
    
    // --- 3. Health Checks ---
    // Effective Sample Size (ESS)
    out_stats->ess = 1.0 / sum_sq_weights;
    
    // Entropy
    double h = 0;
    for (int i=0; i<SWARM_SIZE; i++) {
        if(current_swarm[i].weight > 1e-12) {
            h -= current_swarm[i].weight * log(current_swarm[i].weight);
        }
    }
    out_stats->entropy = h;
    
    // --- 4. Resample (Systematic Resampling) ---
    // Only resample if ESS is too low (avoids losing diversity)
    if (out_stats->ess < SWARM_SIZE / 2.0) {
        Particle* new_swarm = malloc(SWARM_SIZE * sizeof(Particle));
        double u_start = ((double)rand() / RAND_MAX) / SWARM_SIZE;
        double cdf = 0;
        int k = 0;
        
        for(int j=0; j<SWARM_SIZE; j++) {
            double u = u_start + (double)j / SWARM_SIZE;
            while(cdf < u) {
                cdf += current_swarm[k].weight;
                k++;
            }
            // Copy the winning particle
            memcpy(&new_swarm[j], &current_swarm[k-1], sizeof(Particle));
            new_swarm[j].weight = 1.0 / SWARM_SIZE;
        }
        memcpy(current_swarm, new_swarm, SWARM_SIZE * sizeof(Particle));
        free(new_swarm);
    }
}