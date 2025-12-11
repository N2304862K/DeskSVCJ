#include "svcj.h"
#include <float.h>

// --- RNG ---
double norm_rand() {
    double u1 = (double)rand() / RAND_MAX;
    if (u1 < 1e-9) u1 = 1e-9;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}
double lognorm_rand(double mu, double std) { return exp(mu + std*norm_rand()); }

// --- Standard Filter Logic ---
void compute_log_returns(double* ohlcv, int n, double* out) {
    for(int i=1; i<n; i++) out[i-1] = log(ohlcv[i*N_COLS+3]/ohlcv[(i-1)*N_COLS+3]);
}

void generate_prior_swarm(double* ohlcv, int n, double dt, Particle* out) {
    srand(time(NULL));
    double sum_sq=0; for(int i=1; i<n; i++) { 
        double r=log(ohlcv[i*N_COLS+3]/ohlcv[(i-1)*N_COLS+3]); sum_sq+=r*r; 
    }
    double rv = (sum_sq/(n-1))/dt;
    
    for(int i=0; i<SWARM_SIZE; i++) {
        out[i].theta = lognorm_rand(log(rv), 0.5);
        out[i].kappa = lognorm_rand(log(3.0), 0.5);
        out[i].sigma_v = lognorm_rand(log(0.5), 0.5);
        out[i].rho = -0.5 + 0.2*norm_rand();
        out[i].lambda_j = lognorm_rand(log(0.5), 0.5);
        out[i].mu_j = 0.0; out[i].sigma_j = sqrt(rv);
        out[i].v = out[i].theta; out[i].weight = 1.0/SWARM_SIZE;
    }
}

void run_particle_filter_step(Particle* sw, double ret, double dt, FilterStats* out) {
    double sum_w=0, sum_sq=0, m_v=0, m_z=0;
    
    for(int i=0; i<SWARM_SIZE; i++) {
        Particle* p = &sw[i];
        double vp = p->v + p->kappa*(p->theta - p->v)*dt; if(vp<1e-9)vp=1e-9;
        double y = ret + 0.5*vp*dt;
        double S = vp*dt + p->lambda_j*p->sigma_j*p->sigma_j*dt;
        double ll = (1.0/sqrt(2*M_PI*S)) * exp(-0.5*y*y/S);
        p->weight *= ll; sum_w += p->weight;
        p->v = vp + p->sigma_v*sqrt(vp*dt)*norm_rand(); if(p->v<1e-9)p->v=1e-9;
    }
    
    if(sum_w < 1e-40) { 
        for(int i=0; i<SWARM_SIZE; i++) sw[i].weight = 1.0/SWARM_SIZE; sum_w=1.0;
    }
    
    double entropy = 0;
    for(int i=0; i<SWARM_SIZE; i++) {
        sw[i].weight /= sum_w;
        sum_sq += sw[i].weight*sw[i].weight;
        m_v += sw[i].weight * sqrt(sw[i].v);
        double tot = sw[i].v + sw[i].lambda_j*sw[i].sigma_j*sw[i].sigma_j;
        m_z += sw[i].weight * (ret / sqrt(tot*dt));
        if(sw[i].weight > 1e-12) entropy -= sw[i].weight*log(sw[i].weight);
    }
    out->spot_vol_mean = m_v;
    out->innovation_z = m_z;
    out->ess = 1.0/sum_sq;
    out->entropy = entropy;
    
    if(out->ess < SWARM_SIZE/4) {
        Particle* nw = malloc(SWARM_SIZE*sizeof(Particle));
        double c=0, u=((double)rand()/RAND_MAX)/SWARM_SIZE; int k=0;
        for(int i=0; i<SWARM_SIZE; i++) {
            double t = u + (double)i/SWARM_SIZE;
            while(c<t) c+=sw[k++].weight;
            nw[i]=sw[k-1]; nw[i].weight=1.0/SWARM_SIZE;
        }
        memcpy(sw, nw, SWARM_SIZE*sizeof(Particle)); free(nw);
    }
}

// --- NEW: ASYMMETRIC SIMULATION ENGINE ---
void run_asymmetric_simulation(Particle* swarm, double price, double z_score, double dt, MarketMicrostructure m, ContrastiveResult* out) {
    Particle scenarios[SIM_SCENARIOS];
    double u = ((double)rand()/RAND_MAX)/SIM_SCENARIOS; double c=0; int k=0;
    for(int i=0; i<SIM_SCENARIOS; i++) {
        double t = u + (double)i/SIM_SCENARIOS;
        while(c<t && k<SWARM_SIZE) c+=swarm[k++].weight;
        scenarios[i] = swarm[k-1];
    }
    
    double s_l=0, s2_l=0, s_s=0, s2_s=0, s_h=0, s_fric=0;
    int n_sims = SIM_SCENARIOS * PATHS_PER_SCENARIO;
    
    // --- PHYSICS POLARIZATION ---
    // Bias physics based on Z-Score magnitude
    // If Z > 0, Up Jumps are more likely.
    double bias_up = (z_score > 0) ? (1.0 + 0.3*fabs(z_score)) : (1.0 / (1.0 + 0.3*fabs(z_score)));
    double bias_down = 1.0 / bias_up;
    
    for(int s=0; s<SIM_SCENARIOS; s++) {
        Particle p = scenarios[s];
        double current_vol = sqrt(p.v);
        
        // Directional Friction: Cheaper to trade WITH momentum
        // Z > 0 (Up) -> Long cost low, Short cost high
        double cost_long = price * (m.spread_bps + m.impact_coef * current_vol * (z_score > 0 ? 0.7 : 1.3));
        double cost_short = price * (m.spread_bps + m.impact_coef * current_vol * (z_score < 0 ? 0.7 : 1.3));
        
        // Track avg friction for reporting
        s_fric += (cost_long + cost_short) / 2.0;
        
        double stop_dist = m.stop_sigma * current_vol * sqrt(dt) * price;
        
        for(int n=0; n<PATHS_PER_SCENARIO; n++) {
            double p_curr = price;
            double v_curr = p.v;
            double pnl_l = 0, pnl_s = 0;
            int closed_l = 0, closed_s = 0;
            
            for(int t=0; t<m.horizon; t++) {
                // Conditional Volatility (Feedback Loop)
                // If price moves WITH Z-Score, dampen Sigma (Trend Stability)
                // If price moves AGAINST Z-Score, amplify Sigma (Turbulence)
                double move_dir = (p_curr - price);
                double regime_stab = (move_dir * z_score > 0) ? 0.5 : 1.5; 
                
                double dW = norm_rand();
                v_curr += p.kappa*(p.theta - v_curr)*dt + (p.sigma_v * regime_stab)*sqrt(v_curr*dt)*norm_rand();
                if(v_curr<1e-9) v_curr=1e-9;
                
                // Asymmetric Jumps
                double jump = 0;
                if (((double)rand()/RAND_MAX) < p.lambda_j * bias_up * dt) jump += (0.02 + 0.05*norm_rand());
                if (((double)rand()/RAND_MAX) < p.lambda_j * bias_down * dt) jump -= (0.02 + 0.05*norm_rand());
                
                p_curr *= (1.0 + sqrt(v_curr*dt)*dW + jump);
                
                // Decay
                double decay = exp(-m.decay_rate * t);
                double target = m.target_sigma_init * decay * current_vol * sqrt(dt) * price;
                
                double diff = p_curr - price;
                if(!closed_l) {
                    if(diff < -stop_dist) { pnl_l = -stop_dist - cost_long; closed_l=1; }
                    if(diff > target) { pnl_l = target - cost_long; closed_l=1; }
                }
                if(!closed_s) {
                    if(diff > stop_dist) { pnl_s = -stop_dist - cost_short; closed_s=1; }
                    if(diff < -target) { pnl_s = target - cost_short; closed_s=1; }
                }
                if(closed_l && closed_s) break;
            }
            
            if(!closed_l) pnl_l = (p_curr - price) - cost_long;
            if(!closed_s) pnl_s = (price - p_curr) - cost_short;
            double pnl_h = (p_curr - price); 
            
            s_l += pnl_l; s2_l += pnl_l*pnl_l;
            s_s += pnl_s; s2_s += pnl_s*pnl_s;
            s_h += pnl_h; 
        }
    }
    
    double mu_l = s_l / n_sims;
    double std_l = sqrt((s2_l/n_sims) - mu_l*mu_l);
    
    double mu_s = s_s / n_sims;
    double std_s = sqrt((s2_s/n_sims) - mu_s*mu_s);
    
    double mu_h = s_h / n_sims;
    
    out->friction_cost_avg = s_fric / SIM_SCENARIOS;
    out->alpha_long = mu_l - mu_h; 
    out->alpha_short = mu_s - (-mu_h); 
    
    out->t_stat_long = (std_l > 1e-9) ? (mu_l / (std_l/sqrt(n_sims))) : 0;
    out->t_stat_short = (std_s > 1e-9) ? (mu_s / (std_s/sqrt(n_sims))) : 0;
    
    double pooled_std = sqrt((std_l*std_l + std_s*std_s)/2.0);
    out->cohens_d = (pooled_std > 1e-9) ? fabs(mu_l - mu_s) / pooled_std : 0;
}