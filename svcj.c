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
        swarm[i].v = phys->theta * (0.8 + 0.4 * ((next_rand()%1000)/1000.0));
        swarm[i].mu = 0.0;
        swarm[i].rho = -0.5 + 0.1 * rand_normal(); 
        swarm[i].weight = 1.0 / N_PARTICLES;
        swarm[i].last_log_p = log_p;
    }
}

double calc_range_likelihood(double h, double l, double v, double dt) {
    double range = log(h/l);
    double safe_v = (v < 1e-6) ? 1e-6 : v;
    double expected = 1.66 * sqrt(safe_v * dt);
    double diff = range - expected;
    double var_est = safe_v * dt * 0.5;
    if(var_est < 1e-12) var_est = 1e-12;
    double m_dist = (diff*diff) / var_est;
    if (m_dist > 16.0) return 1e-20; 
    return exp(-1.0 * m_dist);
}

void update_swarm_learning(Particle* swarm, PhysicsParams* phys, 
                           double o, double h, double l, double c, 
                           double vol_ratio, double diurnal_factor, double dt, 
                           SwarmState* out) 
{
    double sum_weight = 0.0;
    double sum_sq_weight = 0.0;
    double log_c = log(c);
    
    // Safety Limits
    if (vol_ratio < 0.1) vol_ratio = 0.1;
    if (vol_ratio > 10.0) vol_ratio = 10.0;
    
    double dt_eff = dt * vol_ratio;
    double theta_eff = phys->theta * diurnal_factor;

    // --- A. PROPAGATE & LEARN ---
    for(int i=0; i<N_PARTICLES; i++) {
        Particle* p = &swarm[i];
        
        // 1. Calculate Prediction BEFORE Update
        // This is the 'Prior' estimate of where price should be
        double pred_p = p->last_log_p + (p->mu - 0.5*p->v)*dt_eff;
        double innov = log_c - pred_p; // The "Surprise"
        
        // 2. PARTICLE LEARNING (The Fix for IC)
        // If the price surprised us UP (innov > 0), the Drift was likely too low.
        // We nudge the Drift UP immediately.
        // Gain = Learning Rate * (Vol-Adjusted)
        // This creates "Inertia" - particles align with the move.
        
        double learn_gain = ADAPTIVE_LEARNING_RATE; 
        p->mu += learn_gain * innov; // Direct Feedback Loop
        
        // Clamp Drift to prevent explosion (2000% APR limits)
        if(p->mu > 20.0) p->mu = 20.0; 
        if(p->mu < -20.0) p->mu = -20.0;
        
        // 3. Standard SVCJ Evolution (Post-Learning)
        double dw_v = rand_normal();
        
        // Heston
        p->v += phys->kappa * (theta_eff - p->v) * dt_eff + phys->sigma_v * sqrt(p->v) * dw_v * sqrt(dt_eff);
        if(p->v < 1e-6) p->v = 1e-6;
        if(p->v > 20.0) p->v = 20.0;
        
        int is_jump = 0;
        if ((next_rand()%10000)/10000.0 < (phys->lambda_j * dt_eff)) is_jump = 1;
        
        // 4. Calculate Likelihood (Posterior)
        // We re-evaluate the likelihood based on the UPDATED state if desired, 
        // but standard SIR uses the prediction. 
        // To crush entropy, we stick to the Innovation magnitude.
        
        double var_tot = p->v*dt_eff + (is_jump ? (phys->mu_j*phys->mu_j + phys->sigma_j*phys->sigma_j) : 0);
        if(var_tot < 1e-9) var_tot = 1e-9;
        
        double m_dist = (innov * innov) / var_tot;
        double like_ret = (m_dist > 12.0) ? 1e-25 : (1.0/sqrt(2*M_PI*var_tot)) * exp(-0.5 * m_dist);
        double like_rng = calc_range_likelihood(h, l, p->v, dt_eff);
        
        // 5. Contrast Boosting (Entropy Crusher)
        // Square the likelihood to punish disagreement
        p->weight *= (like_ret * like_rng) * (like_ret * like_rng);
        
        p->last_log_p = log_c;
        if(p->weight < 1e-150) p->weight = 1e-150;
        sum_weight += p->weight;
    }
    
    // --- B. LAZARUS RESET ---
    int collapsed = 0;
    if(sum_weight < 1e-100 || isnan(sum_weight)) {
        collapsed = 1;
        sum_weight = 1.0;
        for(int i=0; i<N_PARTICLES; i++) {
            swarm[i].weight = 1.0 / N_PARTICLES;
            swarm[i].v = theta_eff;
            swarm[i].mu = 0; // Reset drift on collapse
        }
    }
    out->collapsed = collapsed;

    // --- C. AGGREGATE ---
    double entropy = 0;
    out->ev_vol = 0;
    out->ev_drift = 0;
    
    #define BINS 50
    double bins[BINS] = {0};
    double max_v = theta_eff * 5.0;
    
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
    
    int best_b = 0;
    for(int i=1; i<BINS; i++) if(bins[i] > bins[best_b]) best_b = i;
    out->mode_vol = sqrt(((double)best_b/BINS)*max_v + (0.5/BINS)*max_v);
    
    // --- D. RESAMPLE ---
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
            
            // Minimal Jitter to preserve the "Learned" Drift
            new_s[m].v *= (0.995 + 0.01 * ((next_rand()%1000)/1000.0));
            new_s[m].mu += 0.01 * rand_normal(); // Tiny jitter
            new_s[m].rho = swarm[i].rho;
        }
        memcpy(swarm, new_s, N_PARTICLES * sizeof(Particle));
        free(new_s);
    }
}