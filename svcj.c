#include "svcj.h"
#include <float.h>

// --- RNG Helpers ---
double norm_rand() {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}
double lognorm_rand(double mu, double std) { return exp(mu + std*norm_rand()); }

// --- Existing Filter Logic (Condensed for Brevity) ---
void compute_log_returns(double* ohlcv, int n, double* out) {
    for(int i=1; i<n; i++) out[i-1] = log(ohlcv[i*N_COLS+3]/ohlcv[(i-1)*N_COLS+3]);
}
void get_anchor(double* ohlcv, int n, double dt, double* p, double* e) {
    p[0]=3.0; p[1]=0.04; p[2]=0.5; p[3]=-0.5; p[4]=0.5;
    e[0]=0.3; e[1]=0.004; e[2]=0.05; e[3]=0.05; e[4]=0.05;
}
void generate_prior_swarm(double* ohlcv, int n, double dt, Particle* out) {
    double c[5], e[5]; get_anchor(ohlcv, n, dt, c, e);
    for(int i=0; i<SWARM_SIZE; i++) {
        out[i].kappa=lognorm_rand(log(c[0]), e[0]); out[i].theta=lognorm_rand(log(c[1]), e[1]);
        out[i].sigma_v=lognorm_rand(log(c[2]), e[2]); out[i].rho=c[3]+e[3]*norm_rand();
        out[i].lambda_j=lognorm_rand(log(c[4]), e[4]); out[i].v=out[i].theta; out[i].weight=1.0/SWARM_SIZE;
    }
}
void run_particle_filter_step(Particle* sw, double ret, double dt, FilterStats* out) {
    double sum_w=0, sum_sq_w=0, m_v=0, m2_v=0, m_z=0;
    for(int i=0; i<SWARM_SIZE; i++) {
        Particle* p = &sw[i];
        double vp = p->v + p->kappa*(p->theta - p->v)*dt; if(vp<1e-9)vp=1e-9;
        double y = ret - (-0.5*vp)*dt;
        double S = vp*dt + (p->lambda_j*0.01*dt);
        double ll = (1.0/sqrt(2*M_PI*S))*exp(-0.5*y*y/S);
        p->weight *= ll; sum_w += p->weight;
        p->v = vp + p->sigma_v*sqrt(vp*dt)*norm_rand(); if(p->v<1e-9)p->v=1e-9;
    }
    if(sum_w<1e-30) { for(int i=0; i<SWARM_SIZE; i++) sw[i].weight=1.0/SWARM_SIZE; sum_w=1.0; }
    for(int i=0; i<SWARM_SIZE; i++) {
        sw[i].weight /= sum_w; sum_sq_w += sw[i].weight*sw[i].weight;
        m_v += sw[i].weight*sqrt(sw[i].v);
        m_z += sw[i].weight*(ret/sqrt((sw[i].v+sw[i].lambda_j*0.01)*dt));
    }
    out->spot_vol_mean=m_v; out->innovation_z=m_z; out->ess=1.0/sum_sq_w;
    
    // Resample if needed
    if(out->ess < SWARM_SIZE/2.0) {
        Particle* nw = malloc(SWARM_SIZE*sizeof(Particle));
        double u = ((double)rand()/RAND_MAX)/SWARM_SIZE, c=0; int k=0;
        for(int j=0; j<SWARM_SIZE; j++) {
            double target = u + (double)j/SWARM_SIZE;
            while(c < target) c += sw[k++].weight;
            nw[j] = sw[k-1]; nw[j].weight = 1.0/SWARM_SIZE;
        }
        memcpy(sw, nw, SWARM_SIZE*sizeof(Particle)); free(nw);
    }
}

// --- NEW: SIMULATION ENGINE ---

void run_ev_simulation(Particle* swarm, double current_price, double dt, ActionProfile action, SimulationResult* out) {
    // 1. Resample Representatives (Compression)
    // We pick SIM_SCENARIOS particles based on weight to represent the belief distribution
    Particle scenarios[SIM_SCENARIOS];
    double u = ((double)rand()/RAND_MAX) / SIM_SCENARIOS;
    double cdf = 0;
    int k = 0;
    
    for(int i=0; i<SIM_SCENARIOS; i++) {
        double target = u + (double)i / SIM_SCENARIOS;
        while(cdf < target && k < SWARM_SIZE) {
            cdf += swarm[k].weight;
            k++;
        }
        scenarios[i] = swarm[k-1]; // Copy physics
    }
    
    // 2. Project Paths & Evaluate
    double total_pnl = 0;
    double total_sq_pnl = 0;
    int wins = 0;
    int total_paths = SIM_SCENARIOS * PATHS_PER_SCENARIO;
    
    for(int s=0; s<SIM_SCENARIOS; s++) {
        Particle p = scenarios[s];
        
        // Pre-calculate physical stop distance
        double spot_sigma = sqrt(p.v);
        double stop_dist = action.stop_sigma * spot_sigma * sqrt(dt) * current_price;
        double target_dist = action.target_sigma * spot_sigma * sqrt(dt) * current_price;
        
        for(int path=0; path<PATHS_PER_SCENARIO; path++) {
            double price = current_price;
            double v = p.v;
            double pnl = 0;
            int closed = 0;
            
            // Time Loop
            for(int t=0; t<action.horizon_bars; t++) {
                // Heston Update
                v = v + p.kappa*(p.theta - v)*dt + p.sigma_v*sqrt(v*dt)*norm_rand();
                if(v < 1e-9) v = 1e-9;
                
                // Price Update (Drift=0 for risk neutral, or small residual bias)
                // dS = S * (0 + sqrt(v)*dW + J)
                double diffusion = sqrt(v*dt) * norm_rand();
                
                // Jump?
                double jump = 0;
                if( ((double)rand()/RAND_MAX) < (p.lambda_j * dt) ) {
                    // Jump Size: Normal(-0.05, 0.05) approx
                    jump = -0.05 + 0.05*norm_rand(); 
                }
                
                price *= (1.0 + diffusion + jump);
                
                // Check Action Logic
                double diff = price - current_price;
                if(action.direction == 1) { // LONG
                    if(diff < -stop_dist) { pnl = -stop_dist; closed=1; break; }
                    if(diff > target_dist) { pnl = target_dist; closed=1; break; }
                } else { // SHORT
                    if(diff > stop_dist) { pnl = -stop_dist; closed=1; break; }
                    if(diff < -target_dist) { pnl = target_dist; closed=1; break; }
                }
            }
            
            // If time expired
            if(!closed) {
                pnl = (price - current_price) * action.direction;
            }
            
            // Accumulate Stats
            total_pnl += pnl;
            total_sq_pnl += pnl*pnl;
            if(pnl > 0) wins++;
        }
    }
    
    // 3. Final Statistics
    double ev = total_pnl / total_paths;
    double variance = (total_sq_pnl / total_paths) - (ev * ev);
    double std_dev = sqrt(variance);
    
    out->ev = ev;
    out->std_dev = std_dev;
    out->win_rate = (double)wins / total_paths;
    
    // T-Stat: Signal-to-Noise of the EV
    // Standard Error = StdDev / sqrt(N)
    double stderr = std_dev / sqrt(total_paths);
    out->t_stat = (stderr > 1e-9) ? ev / stderr : 0;
    
    // Kelly: (p*b - q) / b (Simplified)
    // We treat payoff ratio as Target/Stop
    double b = (action.target_sigma / action.stop_sigma);
    double p_win = out->win_rate;
    out->kelly_q = (p_win * b - (1.0 - p_win)) / b;
}