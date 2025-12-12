#include "svcj.hpp"
#include <random>
#include <numeric>
#include <algorithm>
#include <vector>

// --- Lookup Table (LUT) for exp() ---
#define LUT_SIZE 20000
#define LUT_SCALE 1000.0
#define LUT_OFFSET 10.0
double exp_lut[LUT_SIZE];

void init_lut() {
    for (int i = 0; i < LUT_SIZE; ++i) {
        double val = (double)i / LUT_SCALE - LUT_OFFSET;
        exp_lut[i] = exp(val);
    }
}

double fast_exp(double x) {
    if (x < -LUT_OFFSET || x > LUT_OFFSET) return exp(x); // Fallback
    int index = (int)((x + LUT_OFFSET) * LUT_SCALE);
    return exp_lut[index];
}


// --- Particle Filter Step ---
// This runs one step of the particle filter for a SINGLE regime
void particle_filter_step(
    double r, double dt, SVCJParams* p, Particle* particles) 
{
    std::default_random_engine gen;
    
    double total_weight = 0.0;

    // 1. Predict (Propagate particles) & 2. Weight
    for (int i = 0; i < N_PARTICLES; ++i) {
        double v_prev = particles[i].v;
        
        // Predict with noise
        std::normal_distribution<double> process_noise(0.0, p->sigma_v * sqrt(dt));
        double v_pred = v_prev + p->kappa * (p->theta - v_prev) * dt + process_noise(gen);
        if (v_pred < 1e-9) v_pred = 1e-9;
        particles[i].v = v_pred;
        
        // Calculate Likelihood (Weight) for this particle
        double y = r - (p->mu - 0.5 * v_pred) * dt;
        double var_j = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
        double total_var = (v_pred + var_j) * dt;
        if(total_var < 1e-12) total_var = 1e-12;
        
        double likelihood = (1.0 / sqrt(total_var * 2 * M_PI)) * fast_exp(-0.5 * y*y / total_var);
        particles[i].w = likelihood;
        total_weight += likelihood;
    }

    // Normalize weights
    if (total_weight > 1e-12) {
        for (int i = 0; i < N_PARTICLES; ++i) {
            particles[i].w /= total_weight;
        }
    }
    
    // 3. Resample (Systematic Resampling)
    std::vector<Particle> new_particles;
    new_particles.reserve(N_PARTICLES);
    
    double step = 1.0 / N_PARTICLES;
    std::uniform_real_distribution<double> dist(0.0, step);
    double u = dist(gen);
    
    double c = particles[0].w;
    int i = 0;
    
    for (int j = 0; j < N_PARTICLES; ++j) {
        while (u > c) {
            i++;
            c += particles[i].w;
        }
        new_particles.push_back(particles[i]);
        u += step;
    }
    
    // Copy back
    for (int k = 0; k < N_PARTICLES; ++k) {
        particles[k] = new_particles[k];
    }
}


// --- HMM Forward Pass (C++ Version) ---
extern "C" void run_hmm_forward_pass_cpp(
    double return_val, double dt,
    SVCJParams* params_array,
    double* trans_mat,
    double* last_probs,
    Particle** particle_clouds, // Array of pointers to particle clouds
    HMMState* out_state) 
{
    // 1. Prediction Step
    double pred_probs[N_REGIMES] = {0};
    for(int j=0; j<N_REGIMES; j++) {
        for(int i=0; i<N_REGIMES; i++) {
            pred_probs[j] += trans_mat[i*N_REGIMES + j] * last_probs[i];
        }
    }

    // 2. Observation (Run Particle Filters in Parallel)
    double likelihoods[N_REGIMES];
    for(int i=0; i<N_REGIMES; i++) {
        particle_filter_step(return_val, dt, &params_array[i], particle_clouds[i]);
        
        // Aggregate likelihood is the average weight
        double avg_l = 0.0;
        for(int k=0; k<N_PARTICLES; k++) avg_l += particle_clouds[i][k].w;
        likelihoods[i] = avg_l / N_PARTICLES;
    }
    
    // 3. Update Beliefs
    double total_prob = 0;
    for(int i=0; i<N_REGIMES; i++) {
        out_state->probabilities[i] = pred_probs[i] * likelihoods[i];
        total_prob += out_state->probabilities[i];
    }
    
    if (total_prob > 1e-12) {
        for(int i=0; i<N_REGIMES; i++) {
            out_state->probabilities[i] /= total_prob;
        }
    }
    
    // 4. Generate Output Distribution (Histogram)
    // Find the weighted average spot vol across all regimes
    double total_spot_vol = 0;
    for(int i=0; i<N_REGIMES; i++) {
        double regime_avg_v = 0;
        for(int k=0; k<N_PARTICLES; k++) regime_avg_v += particle_clouds[i][k].v;
        regime_avg_v /= N_PARTICLES;
        total_spot_vol += out_state->probabilities[i] * sqrt(regime_avg_v);
    }
    out_state->expected_spot_vol = total_spot_vol;
    
    // Simple most likely state
    int max_idx = 0;
    for(int i=1; i<N_REGIMES; i++) {
        if(out_state->probabilities[i] > out_state->probabilities[max_idx]) {
            max_idx = i;
        }
    }
    out_state->most_likely_regime = max_idx;
}