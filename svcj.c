#include "svcj.h"

// Xorshift RNG
uint64_t s[2] = {123456789, 987654321};
uint64_t next_rand(void) {
    uint64_t x = s[0];
    uint64_t const y = s[1];
    s[0] = y;
    x ^= x << 23;
    s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return s[1] + y;
}

double rand_normal() {
    uint64_t r = next_rand();
    double u1 = (r & 0xFFFFFFFF) / 4294967296.0;
    double u2 = (r >> 32) / 4294967296.0;
    if(u1 < 1e-9) u1 = 1e-9;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

void init_swarm(PhysicsParams* phys, Particle* swarm, double start_price) {
    double log_p = log(start_price);
    for(int i=0; i<N_PARTICLES; i++) {
        swarm[i].v = phys->theta; 
        swarm[i].mu = 0.0;
        swarm[i].rho = -0.5;
        swarm[i].weight = 1.0 / N_PARTICLES;
        swarm[i].last_log_p = log_p;
    }
}

double calc_range_likelihood(double h, double l, double v, double dt) {
    double range = log(h/l);
    // Expected range for Brownian Motion ~ 1.66 * sigma * sqrt(dt)
    // We add a small epsilon to v to prevent division by zero
    double safe_v = (v < 1e-6) ? 1e-6 : v;
    double expected = 1.66 * sqrt(safe_v * dt);
    double diff = range - expected;
    return exp(-0.5 * (diff*diff) / (safe_v * dt));
}

void update_swarm(Particle* swarm, PhysicsParams* phys, 
                  double o, double h, double l, double c, 
                  double vol_ratio, double diurnal_factor, double dt, 
                  SwarmState* out) 
{
    double sum_weight = 0.0;
    double sum_sq_weight = 0.0;
    double log_c = log(c);
    
    // Safety Clamps for Input Scaling
    double v_rat = (vol_ratio > 10.0) ? 10.0 : ((vol_ratio < 0.1) ? 0.1 : vol_ratio);
    double dt_eff = dt * v_rat;
    double theta_eff = phys->theta * diurnal_factor;

    // --- A. PROPAGATE ---
    for(int i=0; i<N_PARTICLES; i++) {
        Particle* p = &swarm[i];
        
        // Brownians
        double dw_v = rand_normal();
        double dw_p = rand_normal();
        double dw_s = p->rho * dw_v + sqrt(1.0 - p->rho*p->rho) * dw_p;
        
        // Variance Evolution (Heston)
        // Clamp v to prevent explosion/negative
        p->v += phys->kappa * (theta_eff - p->v) * dt_eff + phys->sigma_v * sqrt(p->v) * dw_v * sqrt(dt_eff);
        if(p->v < 1e-6) p->v = 1e-6;
        if(p->v > 20.0) p->v = 20.0; // Max 2000% Vol
        
        // Drift Evolution (Random Walk)
        // Clamp mu to prevent NaN propagation
        p->mu += 2.0 * sqrt(p->v) * rand_normal() * sqrt(dt_eff);
        if(p->mu > 20.0) p->mu = 20.0;
        if(p->mu < -20.0) p->mu = -20.0;
        
        // Jump
        int is_jump = 0;
        if ((next_rand()%10000)/10000.0 < (phys->lambda_j * dt_eff)) is_jump = 1;
        
        // Likelihood
        double pred = p->last_log_p + (p->mu - 0.5*p->v)*dt_eff + (is_jump ? phys->mu_j : 0);
        double innov = log_c - pred;
        double var_tot = p->v*dt_eff + (is_jump ? (phys->mu_j*phys->mu_j + phys->sigma_j*phys->sigma_j) : 0);
        if(var_tot < 1e-9) var_tot = 1e-9;
        
        double like_ret = (1.0/sqrt(2*M_PI*var_tot)) * exp(-0.5*innov*innov/var_tot);
        double like_rng = calc_range_likelihood(h, l, p->v, dt_eff);
        
        // Update Weight (Safety Floor)
        double combined_likelihood = like_ret * like_rng;
        if(isnan(combined_likelihood) || combined_likelihood < 1e-150) combined_likelihood = 1e-150;
        
        p->weight *= combined_likelihood;
        p->last_log_p = log_c; // Update memory
        
        sum_weight += p->weight;
    }
    
    // --- B. LAZARUS RESET (Critical Fix) ---
    // If swarm collapses (sum_weight ~ 0), reset to uniform distribution
    // This handles the "Gap" problem where no particle predicted the move.
    int collapsed = 0;
    if(sum_weight < 1e-100 || isnan(sum_weight)) {
        collapsed = 1;
        sum_weight = 1.0; // Reset sum
        for(int i=0; i<N_PARTICLES; i++) {
            swarm[i].weight = 1.0 / N_PARTICLES;
            // Optional: Pull Drift/Vol back to priors?
            // For now, just re-weighting allows survival
        }
    }
    out->collapse_count = collapsed;

    // --- C. AGGREGATE ---
    double entropy = 0;
    out->ev_vol = 0;
    out->ev_drift = 0;
    
    // Histogram for Mode
    #define BINS 50
    double bins[BINS] = {0};
    double max_v = theta_eff * 10.0;
    
    for(int i=0; i<N_PARTICLES; i++) {
        swarm[i].weight /= sum_weight;
        double w = swarm[i].weight;
        sum_sq_weight += w*w;
        
        if(w > 1e-12) entropy -= w * log(w);
        
        out->ev_vol += w * sqrt(swarm[i].v);
        out->ev_drift += w * swarm[i].mu;
        
        int b = (int)((swarm[i].v / max_v) * BINS);
        if(b >= BINS) b = BINS-1;
        bins[b] += w;
    }
    
    out->entropy = entropy / log(N_PARTICLES);
    if(isnan(out->entropy)) out->entropy = 1.0; // Max uncertainty if NaN
    
    int best_b = 0;
    for(int i=1; i<BINS; i++) if(bins[i] > bins[best_b]) best_b = i;
    out->mode_vol = sqrt(((double)best_b/BINS)*max_v + (0.5/BINS)*max_v);
    
    // --- D. RESAMPLE ---
    if(1.0/sum_sq_weight < MIN_EFFECTIVE_PARTICLES) {
        Particle* new_s = malloc(N_PARTICLES * sizeof(Particle));
        double r = (next_rand()%10000)/10000.0 * (1.0/N_PARTICLES);
        double c_w = swarm[0].weight;
        int i=0;
        
        for(int m=0; m<N_PARTICLES; m++) {
            double u = r + (double)m/N_PARTICLES;
            while(u > c_w && i < N_PARTICLES-1) { i++; c_w += swarm[i].weight; }
            
            new_s[m] = swarm[i];
            new_s[m].weight = 1.0/N_PARTICLES;
            
            // Jitter
            new_s[m].v *= (0.95 + 0.1 * ((next_rand()%1000)/1000.0));
            new_s[m].rho += 0.05 * rand_normal();
            if(new_s[m].rho>0.99) new_s[m].rho=0.99;
            if(new_s[m].rho<-0.99) new_s[m].rho=-0.99;
        }
        memcpy(swarm, new_s, N_PARTICLES * sizeof(Particle));
        free(new_s);
    }
}