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
    int split = N_PARTICLES / 3;
    
    for(int i=0; i<N_PARTICLES; i++) {
        // Shared Init
        swarm[i].v = phys->theta; 
        swarm[i].rho = -0.5;
        swarm[i].weight = 1.0 / N_PARTICLES;
        swarm[i].last_log_p = log_p;
        
        // Regime Splitting
        if (i < split) {
            // BULL SWARM
            swarm[i].regime = 1;
            swarm[i].mu = 0.5; // Positive Bias
        } else if (i < 2*split) {
            // BEAR SWARM
            swarm[i].regime = -1;
            swarm[i].mu = -0.5; // Negative Bias
        } else {
            // NEUTRAL SWARM
            swarm[i].regime = 0;
            swarm[i].mu = 0.0; // Pinned
        }
    }
}

double calc_range_likelihood(double h, double l, double v, double dt) {
    double range = log(h/l);
    double safe_v = (v < 1e-6) ? 1e-6 : v;
    double expected = 1.66 * sqrt(safe_v * dt);
    double diff = range - expected;
    double var_est = safe_v * dt;
    if(var_est < 1e-12) var_est = 1e-12;
    return exp(-1.0 * (diff*diff) / var_est);
}

void update_swarm_regime(Particle* swarm, PhysicsParams* phys, 
                         double o, double h, double l, double c, 
                         double vol_ratio, double diurnal_factor, double dt, 
                         SwarmState* out) 
{
    double sum_weight = 0.0;
    double sum_sq_weight = 0.0;
    double log_c = log(c);
    
    if (vol_ratio < 0.1) vol_ratio = 0.1;
    if (vol_ratio > 10.0) vol_ratio = 10.0;
    
    double dt_eff = dt * vol_ratio;
    double theta_eff = phys->theta * diurnal_factor;

    // --- A. PROPAGATE ---
    for(int i=0; i<N_PARTICLES; i++) {
        Particle* p = &swarm[i];
        
        double dw_v = rand_normal();
        
        // 1. Variance (Heston) - All regimes share this physics
        p->v += phys->kappa * (theta_eff - p->v) * dt_eff + phys->sigma_v * sqrt(p->v) * dw_v * sqrt(dt_eff);
        if(p->v < 1e-6) p->v = 1e-6;
        if(p->v > 20.0) p->v = 20.0;
        
        // 2. Regime-Specific Drift Logic
        if (p->regime == 0) {
            // NEUTRAL: Drift is pinned to 0. Must explain all moves via Noise.
            p->mu = 0.0;
        } 
        else {
            // TREND: Adaptive Learning
            // Calculate innovation based on OLD drift
            double pred_old = p->last_log_p + (p->mu - 0.5*p->v)*dt_eff;
            double innov = log_c - pred_old;
            
            // Learn: If Bull(1) sees positive innov, accelerate.
            // Gain is scaled by sqrt(dt) to be visible against noise
            double learn_rate = 10.0; 
            p->mu += learn_rate * innov;
            
            // Regime Guard Rails: Bulls stay Bullish, Bears stay Bearish
            if (p->regime == 1 && p->mu < 0.0) p->mu = 0.01;
            if (p->regime == -1 && p->mu > 0.0) p->mu = -0.01;
        }
        
        // Clamp Drift
        if(p->mu > 50.0) p->mu = 50.0; if(p->mu < -50.0) p->mu = -50.0;
        
        // 3. Likelihood
        double pred = p->last_log_p + (p->mu - 0.5*p->v)*dt_eff;
        double innov = log_c - pred;
        double var_tot = p->v * dt_eff;
        if(var_tot < 1e-9) var_tot = 1e-9;
        
        double m_dist = (innov * innov) / var_tot;
        double like_ret = (m_dist > 16.0) ? 1e-25 : (1.0/sqrt(2*M_PI*var_tot)) * exp(-0.5 * m_dist);
        double like_rng = calc_range_likelihood(h, l, p->v, dt_eff);
        
        // --- B. VARIANCE PENALTY (The Fix) ---
        // Penalize particles that have high variance.
        // This favors particles that explain the move via DRIFT (Trend) rather than NOISE (Vol).
        // Penalty factor: 1 / (1 + Variance)
        double vol_penalty = 1.0 / (1.0 + p->v * 2.0);
        
        p->weight *= (like_ret * like_rng * vol_penalty);
        p->last_log_p = log_c;
        if(p->weight < 1e-150) p->weight = 1e-150;
        
        sum_weight += p->weight;
    }
    
    // --- C. LAZARUS ---
    int collapsed = 0;
    if(sum_weight < 1e-100 || isnan(sum_weight)) {
        collapsed = 1;
        sum_weight = 1.0;
        int split = N_PARTICLES / 3;
        for(int i=0; i<N_PARTICLES; i++) {
            swarm[i].weight = 1.0 / N_PARTICLES;
            swarm[i].v = theta_eff;
            // Reset Regimes
            if(i < split) { swarm[i].regime = 1; swarm[i].mu = 0.1; }
            else if(i < 2*split) { swarm[i].regime = -1; swarm[i].mu = -0.1; }
            else { swarm[i].regime = 0; swarm[i].mu = 0.0; }
        }
    }
    out->collapsed = collapsed;

    // --- D. AGGREGATE PROBABILITIES ---
    double entropy = 0;
    out->prob_bull = 0;
    out->prob_bear = 0;
    out->prob_neutral = 0;
    
    #define BINS 50
    double bins[BINS] = {0};
    double max_v = theta_eff * 5.0;
    
    for(int i=0; i<N_PARTICLES; i++) {
        swarm[i].weight /= sum_weight;
        double w = swarm[i].weight;
        sum_sq_weight += w*w;
        
        if(w > 1e-12) entropy -= w * log(w);
        
        // Sum Probabilities by Regime
        if(swarm[i].regime == 1) out->prob_bull += w;
        else if(swarm[i].regime == -1) out->prob_bear += w;
        else out->prob_neutral += w;
        
        int b = (int)((swarm[i].v / max_v) * BINS);
        if(b >= BINS) b = BINS-1;
        bins[b] += w;
    }
    
    out->entropy = entropy / log(N_PARTICLES);
    
    int best_b = 0;
    for(int i=1; i<BINS; i++) if(bins[i] > bins[best_b]) best_b = i;
    out->mode_vol = sqrt(((double)best_b/BINS)*max_v + (0.5/BINS)*max_v);
    
    // --- E. RESAMPLE (Regime Preserving) ---
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
            
            // Jitter
            new_s[m].v *= (0.98 + 0.04 * ((next_rand()%1000)/1000.0));
            new_s[m].rho = swarm[i].rho;
            
            // Jitter Drift but respect Regime
            double d_noise = 0.05 * rand_normal();
            new_s[m].mu += d_noise;
            if (new_s[m].regime == 1 && new_s[m].mu < 0) new_s[m].mu = 0.01;
            if (new_s[m].regime == -1 && new_s[m].mu > 0) new_s[m].mu = -0.01;
            if (new_s[m].regime == 0) new_s[m].mu = 0.0;
        }
        memcpy(swarm, new_s, N_PARTICLES * sizeof(Particle));
        free(new_s);
    }
}