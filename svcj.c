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
    double safe_v = (v < 1e-6) ? 1e-6 : v;
    double expected = 1.66 * sqrt(safe_v * dt);
    double diff = range - expected;
    double var_est = safe_v * dt * 0.5; 
    if(var_est < 1e-12) var_est = 1e-12;
    double m_dist = (diff*diff) / var_est;
    if (m_dist > 16.0) return 1e-20; 
    return exp(-1.0 * m_dist);
}

// Sigmoid Contrast Function
// Pushes weights < mean to 0, weights > mean to 1
double sigmoid_weight(double w, double threshold, double sharp) {
    return 1.0 / (1.0 + exp(-sharp * (w - threshold)));
}

void update_swarm(Particle* swarm, PhysicsParams* phys, 
                  double o, double h, double l, double c, 
                  double vol_ratio, double diurnal_factor, double macro_momentum, double dt, 
                  SwarmState* out) 
{
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
        
        // Variance (Heston)
        p->v += phys->kappa * (theta_eff - p->v) * dt_eff + phys->sigma_v * sqrt(p->v) * dw_v * sqrt(dt_eff);
        if(p->v < 1e-6) p->v = 1e-6;
        if(p->v > 20.0) p->v = 20.0;
        
        // Drift (Momentum Injection)
        // Stronger pull to macro trend to reduce drift variance
        double pull = 8.0 * (macro_momentum - p->mu) * dt_eff;
        double noise = 0.5 * sqrt(p->v) * dw_drift * sqrt(dt_eff);
        p->mu += pull + noise;
        
        if(p->mu > 10.0) p->mu = 10.0; if(p->mu < -10.0) p->mu = -10.0;
        
        int is_jump = 0;
        if ((next_rand()%10000)/10000.0 < (phys->lambda_j * dt_eff)) is_jump = 1;
        
        // --- B. RAW LIKELIHOOD ---
        double pred = p->last_log_p + (p->mu - 0.5*p->v)*dt_eff + (is_jump ? phys->mu_j : 0);
        double innov = log_c - pred;
        double var_tot = p->v*dt_eff + (is_jump ? (phys->mu_j*phys->mu_j + phys->sigma_j*phys->sigma_j) : 0);
        if(var_tot < 1e-9) var_tot = 1e-9;
        
        double m_dist = (innov * innov) / var_tot;
        
        double like_ret = (m_dist > 12.0) ? 1e-25 : (1.0/sqrt(2*M_PI*var_tot)) * exp(-0.5 * m_dist);
        double like_rng = calc_range_likelihood(h, l, p->v, dt_eff);
        
        p->weight *= (like_ret * like_rng);
        p->last_log_p = log_c;
        if(p->weight < 1e-150) p->weight = 1e-150;
    }
    
    // --- C. ADAPTIVE ENTROPY CRUSHING (Annealing Loop) ---
    // Goal: Force Entropy < 0.60
    
    double temp = 1.0;
    double entropy = 1.0;
    double sum_weight = 0;
    
    // Max 10 iterations to prevent hang
    for(int k=0; k<10; k++) {
        sum_weight = 0.0;
        
        // Calculate Sum for Norm
        for(int i=0; i<N_PARTICLES; i++) {
            // Apply temperature: W' = W^(1/T)
            // As T -> 0, this acts like ArgMax
            if(k > 0) swarm[i].weight = pow(swarm[i].weight, 1.2); // Progressive sharpening
            sum_weight += swarm[i].weight;
        }
        
        // Safety Check
        if(sum_weight < 1e-100 || isnan(sum_weight)) {
            // Collapse detected - Stop
            out->collapsed = 1;
            // Reset
            sum_weight = 1.0;
            for(int i=0; i<N_PARTICLES; i++) swarm[i].weight = 1.0/N_PARTICLES;
            entropy = 1.0;
            break;
        } else {
            out->collapsed = 0;
        }
        
        // Calc Entropy
        entropy = 0.0;
        for(int i=0; i<N_PARTICLES; i++) {
            double p_norm = swarm[i].weight / sum_weight;
            if(p_norm > 1e-12) entropy -= p_norm * log(p_norm);
        }
        entropy /= log(N_PARTICLES);
        
        // Exit Condition: Confident enough
        if(entropy < 0.60) {
            temp = 1.0 / pow(1.2, k); // Track how much we squeezed
            break; 
        }
    }
    
    out->entropy = entropy;
    out->final_temperature = temp;

    // --- D. AGGREGATE ---
    out->ev_vol = 0;
    out->ev_drift = 0;
    double sum_sq_weight = 0;
    
    #define BINS 50
    double bins[BINS] = {0};
    double max_v = theta_eff * 5.0;
    
    for(int i=0; i<N_PARTICLES; i++) {
        swarm[i].weight /= sum_weight;
        double w = swarm[i].weight;
        sum_sq_weight += w*w;
        
        out->ev_vol += w * sqrt(swarm[i].v);
        out->ev_drift += w * swarm[i].mu;
        
        int b = (int)((swarm[i].v / max_v) * BINS);
        if(b >= BINS) b = BINS-1;
        bins[b] += w;
    }
    
    int best_b = 0;
    for(int i=1; i<BINS; i++) if(bins[i] > bins[best_b]) best_b = i;
    out->mode_vol = sqrt(((double)best_b/BINS)*max_v + (0.5/BINS)*max_v);
    
    // --- E. RESAMPLE ---
    if(1.0/sum_sq_weight < (N_PARTICLES * 0.6)) {
        Particle* new_s = malloc(N_PARTICLES * sizeof(Particle));
        double r = (next_rand()%10000)/10000.0 * (1.0/N_PARTICLES);
        double c_w = swarm[0].weight;
        int i=0;
        
        for(int m=0; m<N_PARTICLES; m++) {
            double u = r + (double)m/N_PARTICLES;
            while(u > c_w && i < N_PARTICLES-1) { i++; c_w += swarm[i].weight; }
            
            new_s[m] = swarm[i];
            new_s[m].weight = 1.0/N_PARTICLES;
            
            // Jitter dependent on entropy
            // Low Entropy = Low Jitter (Trust the signal)
            // High Entropy = High Jitter (Explore)
            double jitter = (out->entropy < 0.5) ? 0.005 : 0.05;
            
            new_s[m].v *= (1.0 + jitter * rand_normal());
            new_s[m].mu += jitter * rand_normal();
            new_s[m].rho = swarm[i].rho;
        }
        memcpy(swarm, new_s, N_PARTICLES * sizeof(Particle));
        free(new_s);
    }
}