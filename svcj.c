#include "svcj.h"

// --- High-Speed RNG (Xorshift128+) ---
// Essential for Particle Filters to avoid rand() bottlenecks
uint64_t s[2] = {123456789, 987654321};

uint64_t next_rand(void) {
    uint64_t x = s[0];
    uint64_t const y = s[1];
    s[0] = y;
    x ^= x << 23;
    s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return s[1] + y;
}

// Box-Muller for Normal Dist
double rand_normal() {
    uint64_t r = next_rand();
    double u1 = (r & 0xFFFFFFFF) / 4294967296.0;
    double u2 = (r >> 32) / 4294967296.0;
    if(u1 < 1e-9) u1 = 1e-9;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// --- Core SMC Logic ---

void init_swarm(PhysicsParams* phys, Particle* swarm, double start_price) {
    for(int i=0; i<N_PARTICLES; i++) {
        // Initialize with perturbation around Long-Run Mean
        swarm[i].v = phys->theta * (0.8 + 0.4 * (next_rand() % 1000)/1000.0);
        swarm[i].mu = 0.0; // Start neutral
        swarm[i].rho = -0.5 + 0.2 * rand_normal(); // Stochastic correlation
        swarm[i].weight = 1.0 / N_PARTICLES;
        swarm[i].last_log_p = log(start_price);
    }
}

// The "Parkinson" Range Likelihood
// Solves Aliasing by looking at High-Low, not just Close-Close
double calc_range_likelihood(double h, double l, double v, double dt) {
    double range = log(h/l);
    double expected_range = 1.66 * sqrt(v * dt); // Approx expectation for Brownian Motion
    double diff = range - expected_range;
    return exp(-0.5 * (diff*diff) / (v*dt));
}

void update_swarm(Particle* swarm, PhysicsParams* phys, 
                  double o, double h, double l, double c, 
                  double vol_ratio, double diurnal_factor, double dt, 
                  SwarmState* out) 
{
    double sum_weight = 0.0;
    double sum_sq_weight = 0.0;
    double log_c = log(c);
    
    // 1. Volume Clock & Seasonality Adjustment
    // Time moves faster when volume is high
    double dt_eff = dt * vol_ratio; 
    // Structural Gravity (Theta) shifts based on time of day
    double theta_eff = phys->theta * diurnal_factor;

    // --- A. PROPAGATE & WEIGHT (Parallelizable) ---
    for(int i=0; i<N_PARTICLES; i++) {
        Particle* p = &swarm[i];
        
        // Evolve Variance (Heston with Diurnal Theta)
        double dw_v = rand_normal();
        double dw_p = rand_normal();
        // Correlate the brownian motions
        double dw_s = p->rho * dw_v + sqrt(1.0 - p->rho*p->rho) * dw_p;
        
        // Update Variance
        double v_prev = p->v;
        p->v += phys->kappa * (theta_eff - p->v) * dt_eff + phys->sigma_v * sqrt(p->v) * dw_v * sqrt(dt_eff);
        if(p->v < 1e-9) p->v = 1e-9;
        
        // Update Trend (Drift) - AR(1) process
        // Allows particle to "lock in" to a trend separate from volatility
        p->mu += 5.0 * (0.0 - p->mu) * dt_eff + 0.5 * sqrt(p->v) * rand_normal() * sqrt(dt_eff);
        
        // Check for Jumps (Poisson)
        int is_jump = 0;
        if ((next_rand() % 10000)/10000.0 < (phys->lambda_j * dt_eff)) {
            is_jump = 1;
        }
        
        // Calculate Likelihood (The "Judge")
        // 1. Return Likelihood
        double pred_log_p = p->last_log_p + (p->mu - 0.5*p->v)*dt_eff + (is_jump ? phys->mu_j : 0);
        double ret_innov = log_c - pred_log_p;
        double total_var = p->v * dt_eff + (is_jump ? (phys->mu_j*phys->mu_j + phys->sigma_j*phys->sigma_j) : 0);
        
        double like_ret = (1.0/sqrt(2*M_PI*total_var)) * exp(-0.5 * ret_innov*ret_innov / total_var);
        
        // 2. Range Likelihood (Anti-Aliasing)
        double like_range = calc_range_likelihood(h, l, p->v, dt_eff);
        
        // Combine
        p->weight *= (like_ret * like_range);
        p->last_log_p = log_c;
        
        sum_weight += p->weight;
    }
    
    // --- B. NORMALIZE & AGGREGATE ---
    double entropy = 0.0;
    double w_max = 0.0;
    
    // Histogram for Mode Calculation
    #define BINS 50
    double bins[BINS] = {0};
    double max_v_seen = theta_eff * 5.0; // Dynamic range
    
    out->ev_vol = 0;
    out->ev_drift = 0;
    
    for(int i=0; i<N_PARTICLES; i++) {
        swarm[i].weight /= sum_weight; // Normalize
        double w = swarm[i].weight;
        
        sum_sq_weight += w*w;
        if(w > w_max) w_max = w;
        if(w > 1e-9) entropy -= w * log(w);
        
        // Expected Values
        out->ev_vol += w * sqrt(swarm[i].v);
        out->ev_drift += w * swarm[i].mu;
        
        // Mode Binning
        int bin_idx = (int)((swarm[i].v / max_v_seen) * BINS);
        if(bin_idx >= BINS) bin_idx = BINS-1;
        bins[bin_idx] += w;
    }
    
    out->entropy = entropy / log(N_PARTICLES); // Normalized 0-1
    
    // Find Mode
    int best_bin = 0;
    for(int i=1; i<BINS; i++) if(bins[i] > bins[best_bin]) best_bin = i;
    out->mode_vol = sqrt(((double)best_bin / BINS) * max_v_seen);
    
    // --- C. RESAMPLE (Regularized) ---
    // If effective particles low, resample
    double n_eff = 1.0 / sum_sq_weight;
    
    if(n_eff < MIN_EFFECTIVE_PARTICLES) {
        Particle* new_swarm = malloc(N_PARTICLES * sizeof(Particle));
        
        // Low Variance Resampling
        double r = (next_rand() % 10000) / 10000.0 * (1.0/N_PARTICLES);
        double c = swarm[0].weight;
        int i = 0;
        
        for(int m=0; m<N_PARTICLES; m++) {
            double u = r + (double)m/N_PARTICLES;
            while(u > c && i < N_PARTICLES-1) {
                i++;
                c += swarm[i].weight;
            }
            
            new_swarm[m] = swarm[i];
            new_swarm[m].weight = 1.0/N_PARTICLES;
            
            // Jitter / Regularization (Kernel Density)
            // Prevents particle collapse
            new_swarm[m].v *= (0.95 + 0.1 * ((next_rand()%1000)/1000.0));
            new_swarm[m].rho += 0.05 * rand_normal();
            if(new_swarm[m].rho > 0.99) new_swarm[m].rho = 0.99;
            if(new_swarm[m].rho < -0.99) new_swarm[m].rho = -0.99;
        }
        
        memcpy(swarm, new_swarm, N_PARTICLES * sizeof(Particle));
        free(new_swarm);
    }
}