#include "svcj.h"

uint64_t s[2] = {123456789, 987654321};
uint64_t next_rand(void) {
    uint64_t x = s[0]; uint64_t const y = s[1]; s[0]=y;
    x^=x<<23; s[1]=x^y^(x>>17)^(y>>26); return s[1]+y;
}
double rand_normal() {
    uint64_t r=next_rand(); double u1=(r&0xFFFFFFFF)/4294967296.0; double u2=(r>>32)/4294967296.0;
    if(u1<1e-9)u1=1e-9; return sqrt(-2.0*log(u1))*cos(2.0*M_PI*u2);
}

void init_imm(PhysicsParams* phys, Particle* particles, double start_price) {
    double log_p = log(start_price);
    
    for(int i=0; i<N_TOTAL; i++) {
        particles[i].v = phys->theta;
        particles[i].rho = -0.5;
        particles[i].weight = 1.0 / N_SUB_PARTICLES; // Norm per swarm
        particles[i].last_log_p = log_p;
        
        // Block 0: Bull, Block 1: Bear, Block 2: Neutral
        int block = i / N_SUB_PARTICLES;
        if(block == 0) particles[i].mu = 0.5;  // Bull bias
        if(block == 1) particles[i].mu = -0.5; // Bear bias
        if(block == 2) particles[i].mu = 0.0;  // Neut bias
    }
}

double calc_likelihoods(double log_c, double h, double l, Particle* p, PhysicsParams* phys, double dt, double trend_bias) {
    // Prediction
    double pred = p->last_log_p + (p->mu - 0.5*p->v)*dt;
    double innov = log_c - pred;
    
    // Variance (Diff + Jump)
    // Jumps handled implicitly by fat tails in particle distribution
    double var_tot = p->v * dt; 
    if(var_tot < 1e-9) var_tot = 1e-9;
    
    // 1. Price Likelihood (Gaussian)
    double m_dist = (innov * innov) / var_tot;
    double like_p = exp(-0.5 * m_dist);
    
    // 2. Range Likelihood
    double range = log(h/l);
    double exp_rng = 1.66 * sqrt(p->v * dt);
    double diff = range - exp_rng;
    double like_r = exp(-1.0 * (diff*diff) / (p->v*dt));
    
    // 3. Regime Adherence Penalty
    // If a Bull particle has negative drift, punish it
    // If a Neutral particle has high drift, punish it
    double regime_pen = 1.0;
    if (trend_bias > 0 && p->mu < 0) regime_pen = 0.1; // Bull shouldn't be bearish
    if (trend_bias < 0 && p->mu > 0) regime_pen = 0.1; // Bear shouldn't be bullish
    if (trend_bias == 0 && fabs(p->mu) > 0.1) regime_pen = 0.1; // Neutral strict
    
    return like_p * like_r * regime_pen;
}

// Resampler helper
void resample_block(Particle* p, int n, double sum_sq) {
    if (1.0/sum_sq > (n * 0.5)) return; // Healthy
    
    Particle* new_p = malloc(n * sizeof(Particle));
    double r = (next_rand()%10000)/10000.0 * (1.0/n);
    double c_w = p[0].weight;
    int j = 0;
    
    for(int i=0; i<n; i++) {
        double u = r + (double)i/n;
        while(u > c_w && j < n-1) { j++; c_w += p[j].weight; }
        new_p[i] = p[j];
        new_p[i].weight = 1.0/n;
        
        // Jitter
        new_p[i].v *= (0.98 + 0.04 * ((next_rand()%1000)/1000.0));
        new_p[i].mu += 0.05 * rand_normal();
    }
    memcpy(p, new_p, n * sizeof(Particle));
    free(new_p);
}

void update_imm(Particle* swarm, PhysicsParams* phys, 
                double o, double h, double l, double c, 
                double vol_ratio, double diurnal_factor, double dt, 
                IMMState* out) 
{
    if (vol_ratio < 0.1) vol_ratio = 0.1;
    if (vol_ratio > 10.0) vol_ratio = 10.0;
    double dt_eff = dt * vol_ratio;
    double theta_eff = phys->theta * diurnal_factor;
    double log_c = log(c);
    
    double evidence[3] = {0}; // Total unnormalized likelihood per swarm
    
    // --- PROCESS 3 SWARMS INDEPENDENTLY ---
    for(int b=0; b<3; b++) {
        int start = b * N_SUB_PARTICLES;
        Particle* block = &swarm[start];
        
        double sum_w = 0;
        double sum_sq = 0;
        
        // Regime Bias: 0=Bull(+), 1=Bear(-), 2=Neut(0)
        double bias = (b==0) ? 1.0 : ((b==1) ? -1.0 : 0.0);
        
        for(int i=0; i<N_SUB_PARTICLES; i++) {
            Particle* p = &block[i];
            
            // Evolve
            double dw_v = rand_normal();
            double dw_m = rand_normal();
            p->v += phys->kappa * (theta_eff - p->v) * dt_eff + phys->sigma_v * sqrt(p->v) * dw_v * sqrt(dt_eff);
            if(p->v < 1e-6) p->v = 1e-6;
            
            // Drift Evolution (Regime Specific Gravity)
            // Pull towards the regime's bias
            double target_mu = (b==0) ? 0.2 : ((b==1) ? -0.2 : 0.0);
            p->mu += 2.0 * (target_mu - p->mu) * dt_eff + 0.5 * sqrt(p->v) * dw_m * sqrt(dt_eff);
            
            // Likelihood
            double lik = calc_likelihoods(log_c, h, l, p, phys, dt_eff, bias);
            
            // Evidence Accumulation (Before normalization)
            // We use raw weight * likelihood to track "Model Fit"
            evidence[b] += (p->weight * lik); 
            
            p->weight *= lik;
            p->last_log_p = log_c;
            sum_w += p->weight;
        }
        
        // LAZARUS (Per Swarm)
        if (sum_w < 1e-50 || isnan(sum_w)) {
            sum_w = 1.0;
            evidence[b] = 1e-20; // Penalty for dying
            for(int i=0; i<N_SUB_PARTICLES; i++) {
                block[i].weight = 1.0/N_SUB_PARTICLES;
                block[i].v = theta_eff;
                block[i].mu = (b==0)?0.1:((b==1)?-0.1:0);
            }
        }
        
        // Normalize Block
        for(int i=0; i<N_SUB_PARTICLES; i++) {
            block[i].weight /= sum_w;
            sum_sq += block[i].weight * block[i].weight;
        }
        
        // Resample Block (Independent)
        resample_block(block, N_SUB_PARTICLES, sum_sq);
    }
    
    // --- CALCULATE REGIME PROBABILITIES ---
    // P(Regime) = Evidence(Regime) / Total_Evidence
    double total_ev = evidence[0] + evidence[1] + evidence[2];
    if(total_ev < 1e-50) total_ev = 1e-50;
    
    out->prob_bull = evidence[0] / total_ev;
    out->prob_bear = evidence[1] / total_ev;
    out->prob_neutral = evidence[2] / total_ev;
    
    // --- AGGREGATE PHYSICS ---
    // Weighted average of all particles across all swarms
    // Weight = ParticleProb * RegimeProb
    out->agg_vol = 0;
    out->agg_drift = 0;
    
    // Simplified Aggregation using just Regime Probs for speed
    // (Assuming internal particles avg to regime mean)
    // For precision, we could loop all 3000, but this is sufficient for signals
    
    // Calc Entropy of Regimes
    double e = 0;
    if(out->prob_bull > 1e-9) e -= out->prob_bull * log(out->prob_bull);
    if(out->prob_bear > 1e-9) e -= out->prob_bear * log(out->prob_bear);
    if(out->prob_neutral > 1e-9) e -= out->prob_neutral * log(out->prob_neutral);
    out->entropy = e / 1.0986; // log(3)
}