#include "svcj.h"

// Xorshift128+ RNG
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
    for(int i=0; i<N_PARTICLES; i++) {
        // Init Variance around Theta
        swarm[i].v = phys->theta * (0.8 + 0.4 * ((next_rand()%1000)/1000.0));
        swarm[i].mu = 0.0;
        swarm[i].rho = -0.5 + 0.2 * rand_normal();
        swarm[i].weight = 1.0 / N_PARTICLES;
        swarm[i].last_log_p = log(start_price);
    }
}

// Parkinson Range Likelihood (High-Low)
double calc_range_likelihood(double h, double l, double v, double dt) {
    double range = log(h/l);
    double expected = 1.66 * sqrt(v * dt);
    double diff = range - expected;
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
    
    // Volume Clock & Seasonality
    double dt_eff = dt * vol_ratio;
    double theta_eff = phys->theta * diurnal_factor;

    // --- 1. Propagate ---
    for(int i=0; i<N_PARTICLES; i++) {
        Particle* p = &swarm[i];
        
        // Correlated Brownian Motion
        double dw_v = rand_normal();
        double dw_p = rand_normal();
        // p->rho is stochastic per particle
        double dw_s = p->rho * dw_v + sqrt(1.0 - p->rho*p->rho) * dw_p;
        
        // Heston Variance
        p->v += phys->kappa * (theta_eff - p->v) * dt_eff + phys->sigma_v * sqrt(p->v) * dw_v * sqrt(dt_eff);
        if(p->v < 1e-9) p->v = 1e-9;
        
        // Drift (Random Walk) - Allows trend discovery
        // Scale drift noise by Vol to allow adaptation
        p->mu += 2.0 * sqrt(p->v) * rand_normal() * sqrt(dt_eff);
        
        // Jump (Poisson)
        int is_jump = 0;
        if ((next_rand()%10000)/10000.0 < (phys->lambda_j * dt_eff)) is_jump = 1;
        
        // Likelihood (Return)
        double pred_p = p->last_log_p + (p->mu - 0.5*p->v)*dt_eff + (is_jump ? phys->mu_j : 0);
        double innov = log_c - pred_p;
        double var_tot = p->v*dt_eff + (is_jump ? (phys->mu_j*phys->mu_j + phys->sigma_j*phys->sigma_j) : 0);
        
        double like_ret = (1.0/sqrt(2*M_PI*var_tot)) * exp(-0.5*innov*innov/var_tot);
        
        // Likelihood (Range)
        double like_rng = calc_range_likelihood(h, l, p->v, dt_eff);
        
        p->weight *= (like_ret * like_rng);
        p->last_log_p = log_c;
        sum_weight += p->weight;
    }
    
    // --- 2. Aggregate ---
    double entropy = 0;
    out->ev_vol = 0;
    out->ev_drift = 0;
    
    // Mode Histogram
    #define BINS 50
    double bins[BINS] = {0};
    double max_v = theta_eff * 6.0;
    
    for(int i=0; i<N_PARTICLES; i++) {
        swarm[i].weight /= sum_weight;
        double w = swarm[i].weight;
        sum_sq_weight += w*w;
        
        if(w > 1e-9) entropy -= w * log(w);
        
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
    
    // --- 3. Resample ---
    if(1.0/sum_sq_weight < MIN_EFFECTIVE_PARTICLES) {
        Particle* new_swarm = malloc(N_PARTICLES * sizeof(Particle));
        double r = (next_rand()%10000)/10000.0 * (1.0/N_PARTICLES);
        double c = swarm[0].weight;
        int i=0;
        
        for(int m=0; m<N_PARTICLES; m++) {
            double u = r + (double)m/N_PARTICLES;
            while(u > c && i < N_PARTICLES-1) { i++; c += swarm[i].weight; }
            
            new_swarm[m] = swarm[i];
            new_swarm[m].weight = 1.0/N_PARTICLES;
            
            // Jitter to prevent degeneracy
            new_swarm[m].v *= (0.98 + 0.04 * ((next_rand()%1000)/1000.0));
            new_swarm[m].rho += 0.02 * rand_normal();
            if(new_swarm[m].rho>0.99) new_swarm[m].rho=0.99;
            if(new_swarm[m].rho<-0.99) new_swarm[m].rho=-0.99;
        }
        memcpy(swarm, new_swarm, N_PARTICLES * sizeof(Particle));
        free(new_swarm);
    }
}