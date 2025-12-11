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
        // Tighter Init
        swarm[i].v = phys->theta; 
        swarm[i].mu = 0.0;
        swarm[i].rho = -0.5;
        swarm[i].weight = 1.0 / N_PARTICLES;
        swarm[i].last_log_p = log_p;
    }
}

double calc_range_likelihood(double h, double l, double v, double dt) {
    double range = log(h/l);
    // Parkinson Expectation
    double safe_v = (v < 1e-6) ? 1e-6 : v;
    double expected = 1.66 * sqrt(safe_v * dt);
    double diff = range - expected;
    // Sharpen the rejection (Variance of range is prop to V)
    return exp(-1.0 * (diff*diff) / (safe_v * dt)); // Coefficient 1.0 instead of 0.5 for steeper rejection
}

void update_swarm(Particle* swarm, PhysicsParams* phys, 
                  double o, double h, double l, double c, 
                  double vol_ratio, double diurnal_factor, double dt, 
                  SwarmState* out) 
{
    double sum_weight = 0.0;
    double sum_sq_weight = 0.0;
    double log_c = log(c);
    
    // Limits
    if (vol_ratio < 0.1) vol_ratio = 0.1;
    if (vol_ratio > 10.0) vol_ratio = 10.0;
    
    double dt_eff = dt * vol_ratio;
    double theta_eff = phys->theta * diurnal_factor;

    // --- A. PROPAGATE ---
    for(int i=0; i<N_PARTICLES; i++) {
        Particle* p = &swarm[i];
        
        double dw_v = rand_normal();
        double dw_drift = rand_normal();
        
        // 1. Variance Evolution (Heston)
        p->v += phys->kappa * (theta_eff - p->v) * dt_eff + phys->sigma_v * sqrt(p->v) * dw_v * sqrt(dt_eff);
        if(p->v < 1e-6) p->v = 1e-6;
        if(p->v > 10.0) p->v = 10.0;
        
        // 2. Drift Evolution (Random Walk)
        // STRICTER INERTIA: Reduced noise coeff from 2.0 to 0.2
        // This forces particles to commit to a trend line.
        p->mu += 0.2 * sqrt(p->v) * dw_drift * sqrt(dt_eff);
        
        if(p->mu > 10.0) p->mu = 10.0; 
        if(p->mu < -10.0) p->mu = -10.0;
        
        // 3. Jump
        int is_jump = 0;
        if ((next_rand()%10000)/10000.0 < (phys->lambda_j * dt_eff)) is_jump = 1;
        
        // --- B. LIKELIHOOD (With Temperature) ---
        double pred = p->last_log_p + (p->mu - 0.5*p->v)*dt_eff + (is_jump ? phys->mu_j : 0);
        double innov = log_c - pred;
        double var_tot = p->v*dt_eff + (is_jump ? (phys->mu_j*phys->mu_j + phys->sigma_j*phys->sigma_j) : 0);
        if(var_tot < 1e-9) var_tot = 1e-9;
        
        double m_dist = (innov * innov) / var_tot;
        
        // Hard Gating (3 Sigma)
        if (m_dist > 9.0) {
            p->weight = 1e-20;
        } else {
            // Gaussian Likelihood
            double like_ret = exp(-0.5 * m_dist);
            // Range Likelihood
            double like_rng = calc_range_likelihood(h, l, p->v, dt_eff);
            
            // Update
            p->weight *= (like_ret * like_rng);
            
            // CONTRAST ENHANCEMENT (The Fix for Entropy)
            // Square the weights to punish mediocrity
            p->weight = p->weight * p->weight;
        }
        
        p->last_log_p = log_c;
        if(p->weight < 1e-100) p->weight = 1e-100;
        
        sum_weight += p->weight;
    }
    
    // --- C. LAZARUS RESET ---
    int collapsed = 0;
    if(sum_weight < 1e-50 || isnan(sum_weight)) {
        collapsed = 1;
        sum_weight = 1.0;
        for(int i=0; i<N_PARTICLES; i++) {
            swarm[i].weight = 1.0 / N_PARTICLES;
            swarm[i].v = theta_eff; // Reset vol
            swarm[i].mu = 0; // Reset drift
        }
    }
    out->collapsed = collapsed;

    // --- D. AGGREGATE ---
    double entropy = 0;
    out->ev_vol = 0;
    out->ev_drift = 0;
    
    #define BINS 50
    double bins[BINS] = {0};
    double max_v = theta_eff * 5.0;
    
    for(int i=0; i<N_PARTICLES; i++) {
        swarm[i].weight /= sum_weight; // Normalize
        double w = swarm[i].weight;
        sum_sq_weight += w*w;
        
        // Safe Entropy
        if(w > 1e-12) entropy -= w * log(w);
        
        out->ev_vol += w * sqrt(swarm[i].v);
        out->ev_drift += w * swarm[i].mu;
        
        int b = (int)((swarm[i].v / max_v) * BINS);
        if(b >= BINS) b = BINS-1;
        bins[b] += w;
    }
    
    // Theoretical Max Entropy = log(N)
    out->entropy = entropy / log(N_PARTICLES);
    
    // Calculate Mode
    int best_b = 0;
    for(int i=1; i<BINS; i++) if(bins[i] > bins[best_b]) best_b = i;
    out->mode_vol = sqrt(((double)best_b/BINS)*max_v + (0.5/BINS)*max_v);
    
    // --- E. RESAMPLE (Low Jitter) ---
    // Resample if N_eff < 50%
    if(1.0/sum_sq_weight < (N_PARTICLES * 0.5)) {
        Particle* new_s = malloc(N_PARTICLES * sizeof(Particle));
        double r = (next_rand()%10000)/10000.0 * (1.0/N_PARTICLES);
        double c_w = swarm[0].weight;
        int i=0;
        
        for(int m=0; m<N_PARTICLES; m++) {
            double u = r + (double)m/N_PARTICLES;
            while(u > c_w && i < N_PARTICLES-1) { i++; c_w += swarm[i].weight; }
            
            new_s[m] = swarm[i];
            new_s[m].weight = 1.0/N_PARTICLES;
            
            // VERY LOW JITTER (The Fix for Stability)
            // 0.5% Vol Jitter, 0.5% Drift Jitter
            new_s[m].v *= (0.995 + 0.01 * ((next_rand()%1000)/1000.0));
            
            // Micro-jitter on drift to allow evolution without explosion
            new_s[m].mu += 0.01 * rand_normal(); 
            
            new_s[m].rho = swarm[i].rho; // Keep rho stable
        }
        memcpy(swarm, new_s, N_PARTICLES * sizeof(Particle));
        free(new_s);
    }
}