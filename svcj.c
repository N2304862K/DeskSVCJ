#include "svcj_swarm.h"

// --- RNG Helpers ---
// Simple Box-Muller for speed/portability
double get_random_normal() {
    double u = ((double)rand() / RAND_MAX);
    double v = ((double)rand() / RAND_MAX);
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}
double get_random_uniform() { return ((double)rand() / RAND_MAX); }

// --- Initialization ---
void init_swarm(Particle* swarm, SwarmParams* p) {
    for(int i=0; i<N_PARTICLES; i++) {
        swarm[i].v = p->theta;
        swarm[i].mu = 0.0;
        swarm[i].rho = p->rho_mean + 0.1 * get_random_normal(); // Jittered Correlation
        if(swarm[i].rho > 0.99) swarm[i].rho = 0.99;
        if(swarm[i].rho < -0.99) swarm[i].rho = -0.99;
        
        swarm[i].jump_state = 0;
        swarm[i].weight = 1.0 / N_PARTICLES;
    }
}

// --- Step 1: Predict (Evolution) ---
// Solves "Diurnal Seasonality" by scaling drift/diffusion with time-of-day
void predict_step(Particle* swarm, SwarmParams* p, double dt, double diurnal_factor) {
    double sqrt_dt = sqrt(dt);
    
    for(int i=0; i<N_PARTICLES; i++) {
        // 1. Evolve Variance (Heston with Diurnal Scaling)
        // We revert to Theta * DiurnalFactor
        double theta_t = p->theta * diurnal_factor;
        double dw_v = get_random_normal();
        
        double v_prev = swarm[i].v;
        double v_drift = p->kappa * (theta_t - v_prev) * dt;
        double v_diff = p->sigma_v * sqrt(v_prev) * dw_v * sqrt_dt;
        
        swarm[i].v = v_prev + v_drift + v_diff;
        if(swarm[i].v < 1e-6) swarm[i].v = 1e-6; // Floor
        
        // 2. Evolve Correlation (Jacobi Process - Mean Reverting Bounded)
        // Allows system to adapt to "Correlation Breakdown"
        double rho_drift = 2.0 * (p->rho_mean - swarm[i].rho) * dt;
        double rho_diff = 0.2 * sqrt(1 - swarm[i].rho*swarm[i].rho) * get_random_normal() * sqrt_dt;
        swarm[i].rho += rho_drift + rho_diff;
        if(swarm[i].rho > 0.99) swarm[i].rho = 0.99;
        if(swarm[i].rho < -0.99) swarm[i].rho = -0.99;
        
        // 3. Evolve Drift (Local Trend Hypothesis)
        // Random walk for trend perception
        swarm[i].mu += 0.5 * get_random_normal() * sqrt_dt; 
        
        // 4. Jump Switch (Poisson)
        if (get_random_uniform() < p->lambda_j * dt) {
            swarm[i].jump_state = 1;
        } else {
            swarm[i].jump_state = 0;
        }
    }
}

// --- Step 2: Update (Correction) ---
// Solves "Aliasing" by using Range (High-Low) info
void update_step(Particle* swarm, SwarmParams* p, double ret, double range_sq, double dt) {
    double sum_weight = 0.0;
    
    for(int i=0; i<N_PARTICLES; i++) {
        // Expected mean and variance for this particle
        double mu_exp = swarm[i].mu * dt;
        if (swarm[i].jump_state) mu_exp += p->mu_j;
        
        double var_exp = swarm[i].v * dt;
        if (swarm[i].jump_state) var_exp += (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
        
        // 1. Return Likelihood (Gaussian)
        double resid = ret - mu_exp;
        double pdf_ret = (1.0 / sqrt(2 * M_PI * var_exp)) * exp(-0.5 * resid*resid / var_exp);
        
        // 2. Range Likelihood (Parkinson / Aliasing Fix)
        // E[Range^2] approx 4 * ln(2) * var
        // This penalizes particles that predict Low Volatility when Range is high
        double expected_range_sq = 4.0 * 0.693 * var_exp; 
        // Chi-Square like penalty for range mismatch
        double range_ratio = range_sq / (expected_range_sq + 1e-9);
        double pdf_range = exp(-0.5 * (range_ratio + log(expected_range_sq))); 
        
        // Update Weight
        // We combine both evidences.
        swarm[i].weight *= (pdf_ret * pdf_range);
        sum_weight += swarm[i].weight;
    }
    
    // Normalize
    if (sum_weight < 1e-15) sum_weight = 1e-15;
    for(int i=0; i<N_PARTICLES; i++) {
        swarm[i].weight /= sum_weight;
    }
}

// --- Step 3: Resample (Regularized) ---
// Solves "Particle Degeneracy" by jittering clones
void resample_regularized(Particle* swarm, SwarmParams* p) {
    // Calculate Effective Sample Size (ESS)
    double sum_sq_w = 0;
    for(int i=0; i<N_PARTICLES; i++) sum_sq_w += swarm[i].weight * swarm[i].weight;
    double ess = 1.0 / sum_sq_w;
    
    // Only resample if degeneracy is high
    if (ess > N_PARTICLES * 0.5) return;
    
    Particle new_swarm[N_PARTICLES];
    double cdf[N_PARTICLES];
    cdf[0] = swarm[0].weight;
    for(int i=1; i<N_PARTICLES; i++) cdf[i] = cdf[i-1] + swarm[i].weight;
    
    for(int i=0; i<N_PARTICLES; i++) {
        double r = get_random_uniform();
        int idx = 0;
        // Binary search for speed
        int lower = 0, upper = N_PARTICLES-1;
        while(lower <= upper) {
            int mid = lower + (upper-lower)/2;
            if (cdf[mid] < r) lower = mid + 1;
            else { idx = mid; upper = mid - 1; }
        }
        
        new_swarm[i] = swarm[idx];
        new_swarm[i].weight = 1.0 / N_PARTICLES;
        
        // JITTER (Regularization)
        // Prevents collapse into a single point
        new_swarm[i].v *= (1.0 + 0.05 * get_random_normal());
        new_swarm[i].mu += 0.001 * get_random_normal();
    }
    
    memcpy(swarm, new_swarm, N_PARTICLES * sizeof(Particle));
}

// --- Metrics Aggregator ---
void calc_swarm_metrics(Particle* swarm, SwarmMetrics* out) {
    double w_mu = 0;
    double w_rho = 0;
    double w_jump = 0;
    double entropy = 0;
    
    // Sort for Percentile (Risk Vol) - Simplified linear scan for demo
    // Ideally use quickselect for 95th percentile
    // Here we use Weighted Mean + 2 StdDev approximation for speed
    double mean_v = 0;
    double var_v = 0;
    
    for(int i=0; i<N_PARTICLES; i++) {
        double w = swarm[i].weight;
        w_mu += w * swarm[i].mu;
        w_rho += w * swarm[i].rho;
        w_jump += w * swarm[i].jump_state;
        mean_v += w * swarm[i].v;
        
        if (w > 0) entropy -= w * log(w);
    }
    
    for(int i=0; i<N_PARTICLES; i++) {
        var_v += swarm[i].weight * pow(swarm[i].v - mean_v, 2);
    }
    
    out->expected_return = w_mu;
    out->effective_rho = w_rho;
    out->regime_prob = w_jump;
    out->swarm_entropy = entropy;
    
    // Risk Volatility (Upper Bound of Swarm confidence)
    out->risk_volatility = sqrt(mean_v + 2.0 * sqrt(var_v)); 
}