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

void init_swarm(PhysicsParams* phys, Particle* swarm, double start_price, TrendState* ts) {
    double log_p = log(start_price);
    
    // Init Trend Memory
    ts->trend_fast = 0.0;
    ts->trend_mid = 0.0;
    ts->trend_slow = 0.0;
    ts->coherence = 0.0;
    
    for(int i=0; i<N_PARTICLES; i++) {
        swarm[i].v = phys->theta;
        
        // Evolving Physics initialization
        swarm[i].theta = phys->theta * (0.9 + 0.2 * ((next_rand()%1000)/1000.0));
        swarm[i].kappa = phys->kappa * (0.9 + 0.2 * ((next_rand()%1000)/1000.0));
        
        swarm[i].mu = 0.0;
        swarm[i].rho = -0.5 + 0.1 * rand_normal(); 
        swarm[i].weight = 1.0 / N_PARTICLES;
        swarm[i].last_log_p = log_p;
    }
}

// Update Global Multi-Scale Momentum
void update_trends(TrendState* ts, double ret, double dt) {
    // Decays for approx 5m, 15m, 60m response (assuming dt ~ 5min)
    // Adjust decay based on dt (seconds)
    // Alpha = 1 - exp(-dt / Tau)
    // Tau in years... assume dt is annualized.
    // Let's use simple EWMA factors for robustness in this context
    
    double alpha_f = 0.5;  // Fast
    double alpha_m = 0.2;  // Mid
    double alpha_s = 0.05; // Slow
    
    ts->trend_fast += alpha_f * (ret - ts->trend_fast);
    ts->trend_mid  += alpha_m * (ret - ts->trend_mid);
    ts->trend_slow += alpha_s * (ret - ts->trend_slow);
    
    // Coherence: Are they aligned?
    // +1 if all positive, -1 if all negative, 0 if mixed
    double sum_sign = (ts->trend_fast > 0 ? 1 : -1) + 
                      (ts->trend_mid > 0 ? 1 : -1) + 
                      (ts->trend_slow > 0 ? 1 : -1);
    
    // Scaleless Momentum Injection Vector
    // If aligned (abs=3), strong pull. If mixed, weak pull.
    ts->coherence = sum_sign / 3.0; 
}

double calc_range_likelihood(double h, double l, double v, double dt) {
    double range = log(h/l);
    double safe_v = (v < 1e-6) ? 1e-6 : v;
    double expected = 1.66 * sqrt(safe_v * dt);
    double diff = range - expected;
    double var_est = safe_v * dt * 0.5; 
    if(var_est < 1e-12) var_est = 1e-12;
    
    double m_dist = (diff*diff) / var_est;
    if (m_dist > 25.0) return 1e-18; // 5-Sigma 
    return exp(-1.0 * m_dist); 
}

void update_swarm(Particle* swarm, TrendState* ts, PhysicsParams* phys, 
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
    
    // 1. Update Global Momentum State
    // Estimate return from swarm average or just use Close-Close of input?
    // Input C is current close. We need prev close for return.
    // Swarm particles have `last_log_p`. Use the first particle to get prev price approx
    double prev_log_p = swarm[0].last_log_p; // Approximation
    double raw_ret = log_c - prev_log_p;
    
    // Normalize return by dt to get "Drift Intensity"
    double drift_intensity = raw_ret / dt_eff;
    update_trends(ts, drift_intensity, dt_eff);
    
    out->global_trend_coherence = ts->coherence;

    // --- A. PROPAGATE ---
    for(int i=0; i<N_PARTICLES; i++) {
        Particle* p = &swarm[i];
        
        double dw_v = rand_normal();
        double dw_drift = rand_normal();
        
        // 1. Evolve Physics (Parameter Learning)
        // Allow Theta to drift to match regime
        p->theta += 0.5 * (phys->theta - p->theta) * dt_eff + 0.1 * sqrt(p->theta) * rand_normal() * sqrt(dt_eff);
        if(p->theta < 1e-6) p->theta = 1e-6;
        
        // 2. Variance Evolution (Heston with Dynamic Theta)
        p->v += p->kappa * (p->theta * diurnal_factor - p->v) * dt_eff + phys->sigma_v * sqrt(p->v) * dw_v * sqrt(dt_eff);
        if(p->v < 1e-6) p->v = 1e-6;
        if(p->v > 50.0) p->v = 50.0;
        
        // 3. MOMENTUM INJECTION (The Entropy Killer)
        // Pull particle drift towards Global Trend Coherence
        // If Coherence is strong (+1/-1), pull hard. If 0, let random walk dominate.
        double injection_strength = 5.0 * fabs(ts->coherence); 
        
        // Weighted Average of Random Walk and Global Trend
        // p->mu = p->mu + MeanReversionToTrend + Noise
        double target_drift = ts->trend_fast; // Follow fast trend
        
        p->mu += injection_strength * (target_drift - p->mu) * dt_eff + 0.5 * sqrt(p->v) * dw_drift * sqrt(dt_eff);
        
        if(p->mu > 20.0) p->mu = 20.0; if(p->mu < -20.0) p->mu = -20.0;
        
        // 4. Jump
        int is_jump = 0;
        if ((next_rand()%10000)/10000.0 < (phys->lambda_j * dt_eff)) is_jump = 1;
        
        // --- B. LIKELIHOOD ---
        double pred = p->last_log_p + (p->mu - 0.5*p->v)*dt_eff + (is_jump ? phys->mu_j : 0);
        double innov = log_c - pred;
        double var_tot = p->v*dt_eff + (is_jump ? (phys->mu_j*phys->mu_j + phys->sigma_j*phys->sigma_j) : 0);
        if(var_tot < 1e-12) var_tot = 1e-12;
        
        double m_dist = (innov * innov) / var_tot;
        
        double like_ret = 0.0;
        if (m_dist > 25.0) { // 5-Sigma
            like_ret = 1e-25; 
        } else {
            like_ret = (1.0/sqrt(2*M_PI*var_tot)) * exp(-0.5 * m_dist);
        }
        
        double like_rng = calc_range_likelihood(h, l, p->v, dt_eff);
        double raw_w = like_ret * like_rng;
        
        // Contrast Boost (Square weights to kill noise)
        p->weight *= (raw_w * raw_w); 
        
        p->last_log_p = log_c;
        if(p->weight < 1e-150) p->weight = 1e-150;
        
        sum_weight += p->weight;
    }
    
    // --- C. LAZARUS RESET ---
    int collapsed = 0;
    if(sum_weight < 1e-100 || isnan(sum_weight)) {
        collapsed = 1;
        sum_weight = 1.0;
        for(int i=0; i<N_PARTICLES; i++) {
            swarm[i].weight = 1.0 / N_PARTICLES;
            swarm[i].v = phys->theta; // Soft Reset
            swarm[i].mu = ts->trend_fast; // Reset to current trend
        }
    }
    out->collapsed = collapsed;

    // --- D. AGGREGATE ---
    double entropy = 0;
    out->ev_vol = 0;
    out->ev_drift = 0;
    out->avg_theta = 0;
    
    #define BINS 50
    double bins[BINS] = {0};
    double max_v = phys->theta * 8.0;
    
    for(int i=0; i<N_PARTICLES; i++) {
        swarm[i].weight /= sum_weight;
        double w = swarm[i].weight;
        sum_sq_weight += w*w;
        
        if(w > 1e-12) entropy -= w * log(w);
        
        out->ev_vol += w * sqrt(swarm[i].v);
        out->ev_drift += w * swarm[i].mu;
        out->avg_theta += w * swarm[i].theta;
        
        int b = (int)((swarm[i].v / max_v) * BINS);
        if(b >= BINS) b = BINS-1;
        bins[b] += w;
    }
    
    out->entropy = entropy / log(N_PARTICLES);
    
    int best_b = 0;
    for(int i=1; i<BINS; i++) if(bins[i] > bins[best_b]) best_b = i;
    out->mode_vol = sqrt(((double)best_b/BINS)*max_v + (0.5/BINS)*max_v);
    
    // --- E. RESAMPLE (Adaptive Jitter) ---
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
            
            // Adaptive Jitter
            // If Coherence is High, Reduce Jitter (Let it trend)
            // If Coherence is Low, Increase Jitter (Explore)
            double coherence_factor = fabs(ts->coherence);
            double jitter = 0.02 * (1.0 - coherence_factor); 
            if (jitter < 0.001) jitter = 0.001;
            
            new_s[m].v *= (1.0 + jitter * rand_normal());
            new_s[m].mu += jitter * rand_normal(); 
            
            // Evolve Params slowly
            new_s[m].theta *= (1.0 + 0.01 * rand_normal());
            new_s[m].kappa = swarm[i].kappa;
            new_s[m].rho = swarm[i].rho;
        }
        memcpy(swarm, new_s, N_PARTICLES * sizeof(Particle));
        free(new_s);
    }
}