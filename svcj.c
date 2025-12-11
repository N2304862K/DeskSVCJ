#include "svcj.h"
#include <time.h>

// --- Helper: Random Number Generation ---
double rand_norm(double mu, double sigma) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mu + sigma * z;
}

// --- Particle Initialization ---
void init_particle_set(Particle* particles, double theta, double start_price) {
    srand(time(NULL));
    for (int i = 0; i < N_PARTICLES; ++i) {
        particles[i].v = theta; // Start at long-run variance
        particles[i].weight = 1.0 / N_PARTICLES;
        
        // Distribute Regimes
        if (i < N_PARTICLES / 3) {
            particles[i].regime = 0; // Bull
            particles[i].mu = 0.10;  // Positive Drift Bias
        } else if (i < (2 * N_PARTICLES) / 3) {
            particles[i].regime = 1; // Bear
            particles[i].mu = -0.10; // Negative Drift Bias
        } else {
            particles[i].regime = 2; // Neutral
            particles[i].mu = 0.0;   // Zero Drift
        }
    }
}

// --- Core SMC Loop ---
void update_particles(Particle* particles, IMMState* state, double o, double h, double l, double c, double vol_factor, double range_factor) {
    double log_ret = log(c / o);
    double avg_log_likelihood = 0.0;

    // 1. Prediction & Weighting
    for (int i = 0; i < N_PARTICLES; ++i) {
        // A. Drift Decay (Ornstein-Uhlenbeck)
        particles[i].mu *= 0.95; // Decay towards zero
        
        // B. Variance Prediction
        double v_pred = particles[i].v + state->kappa * (state->theta - particles[i].v) * state->dt * vol_factor;
        if (v_pred < 1e-7) v_pred = 1e-7;
        
        // C. Update Variance State
        particles[i].v = v_pred;
        
        // D. Calculate Likelihood
        // Expected mean and variance of this particle's return prediction
        double exp_mu = particles[i].mu * state->dt * vol_factor;
        double exp_sigma = sqrt(v_pred * state->dt * vol_factor) * range_factor;
        if(exp_sigma < 1e-6) exp_sigma = 1e-6;
        
        // Gaussian PDF for Likelihood
        double exponent = -0.5 * pow((log_ret - exp_mu) / exp_sigma, 2);
        double likelihood = (1.0 / (SQRT_2PI * exp_sigma)) * exp(exponent);
        
        particles[i].weight *= likelihood;
        avg_log_likelihood += log(likelihood + 1e-30);
    }
    
    // 2. Regularization & Normalization
    double weight_sum = 0.0;
    
    // B. The "Weight Floor" (Uniform Injection)
    double uniform_w = 0.05 / N_PARTICLES;
    
    for(int i=0; i<N_PARTICLES; i++) {
        // Mix with uniform distribution to keep all particles alive
        particles[i].weight = 0.95 * particles[i].weight + uniform_w;
        weight_sum += particles[i].weight;
    }

    if (weight_sum < 1e-30) { // Safety for underflow
        for(int i=0; i<N_PARTICLES; i++) particles[i].weight = 1.0 / N_PARTICLES;
        weight_sum = 1.0;
    }
    
    // Normalize weights
    double bull_prob = 0, bear_prob = 0, neut_prob = 0;
    for (int i = 0; i < N_PARTICLES; ++i) {
        particles[i].weight /= weight_sum;
        if (particles[i].regime == 0) bull_prob += particles[i].weight;
        if (particles[i].regime == 1) bear_prob += particles[i].weight;
        if (particles[i].regime == 2) neut_prob += particles[i].weight;
    }
    
    // Update State
    state->p_bull = bull_prob;
    state->p_bear = bear_prob;
    state->p_neutral = neut_prob;
    
    // Entropy (Diversity Measure)
    double H = 0;
    if(bull_prob > 0) H -= bull_prob * log2(bull_prob);
    if(bear_prob > 0) H -= bear_prob * log2(bear_prob);
    if(neut_prob > 0) H -= neut_prob * log2(neut_prob);
    state->entropy = H;

    // 3. Resampling (Kill/Clone)
    // Low Variance Resampling for efficiency
    Particle new_particles[N_PARTICLES];
    double r = ((double)rand() / RAND_MAX) / N_PARTICLES;
    double c_w = particles[0].weight;
    int k = 0;
    
    for (int j = 0; j < N_PARTICLES; ++j) {
        double u = r + (double)j / N_PARTICLES;
        while (u > c_w) {
            k++;
            c_w += particles[k].weight;
        }
        
        // Clone particle k
        new_particles[j] = particles[k];
        
        // A. The "Mutation" Step
        double mut_roll = (double)rand() / RAND_MAX;
        if (mut_roll < 0.02) { // 2% chance to mutate
            int current_regime = new_particles[j].regime;
            // Transition: Bull -> Neut, Bear -> Neut, Neut -> Bull/Bear
            if(current_regime == 0 || current_regime == 1) {
                new_particles[j].regime = 2; // Become Neutral
                new_particles[j].mu = 0.0;
            } else {
                new_particles[j].regime = (rand() % 2); // Become Bull or Bear
                new_particles[j].mu = (new_particles[j].regime == 0) ? 0.10 : -0.10;
            }
        }
    }

    // Replace old set with new
    for (int i = 0; i < N_PARTICLES; ++i) {
        particles[i] = new_particles[i];
        particles[i].weight = 1.0 / N_PARTICLES; // Reset weights after resampling
    }
}