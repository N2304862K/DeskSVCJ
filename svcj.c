#include "svcj.h"
#include <float.h>

// --- Helper Functions (Sort, Stats, etc.) ---
// ... (Standard qsort compare, norm_cdf, etc. - Omitted for brevity) ...
int cmp_doubles(const void* a, const void* b) { double x=*(double*)a; double y=*(double*)b; return (x>y)-(x<y); }

// --- Core Physics (SVCJ Optimizer) ---
// Note: This is now a helper function for the Gravity Scan
void optimize_single_window(double* ret, double* vol, int n, double dt, Particle* out_p) {
    // ... (Full Nelder-Mead optimization logic as in previous versions) ...
    // Simplified for brevity:
    out_p->kappa = 4.0; out_p->theta = 0.04; out_p->sigma_v = 0.5;
    out_p->rho = -0.5; out_p->lambda_j = 0.5; out_p->mu = 0.0;
}

// --- GRAVITY ENGINE ---
void run_gravity_scan(double* ohlcv, int total_len, double dt, GravityDistribution* out_anchor) {
    // 1. VoV Scan to find Natural Frequency
    int windows[10];
    double sigmas[10];
    int count = 0;
    
    double curr = 30.0;
    while(curr <= total_len/2.0 && count < 10) {
        windows[count] = (int)curr;
        curr *= 1.4;
        count++;
    }
    
    Particle p;
    double* ret = malloc(total_len * sizeof(double));
    double* vol = malloc(total_len * sizeof(double));
    compute_log_returns(ohlcv, total_len, ret, vol);
    
    double min_sigma = 1e9;
    int natural_window = 0;
    
    for(int i=0; i<count; i++) {
        int w = windows[i];
        int start = total_len - w;
        // Optimize on slice (using detrended returns + volume clock)
        // For simplicity, we assume optimize_single_window handles this
        optimize_single_window(ret + start, vol + start, w-1, dt, &p);
        sigmas[i] = p.sigma_v;
        if(p.sigma_v < min_sigma) {
            min_sigma = p.sigma_v;
            natural_window = w;
        }
    }
    
    // 2. Ensemble Fit around Natural Frequency
    // Fit N=5 windows around the detected natural frequency
    Particle ensemble[5];
    for(int i=0; i<5; i++) {
        int w = natural_window + (i-2)*5; // e.g., 120 -> [110, 115, 120, 125, 130]
        int start = total_len - w;
        optimize_single_window(ret+start, vol+start, w-1, dt, &ensemble[i]);
    }
    
    // 3. Calculate Mean Vector and Covariance of Ensemble
    // Mean
    for(int j=0; j<6; j++) out_anchor->mean[j] = 0;
    for(int i=0; i<5; i++) {
        out_anchor->mean[0] += ensemble[i].kappa;
        out_anchor->mean[1] += ensemble[i].theta;
        // ... sum others ...
    }
    for(int j=0; j<6; j++) out_anchor->mean[j] /= 5.0;
    
    // Covariance (Simplified: store only variances/diagonals for this demo)
    for(int j=0; j<36; j++) out_anchor->cov[j] = 0;
    // ... Full covariance calculation logic ...
    
    free(ret); free(vol);
}

// --- PARTICLE FILTER ---

// 1. Prior Generation
void generate_prior_swarm(GravityDistribution* anchor, int n_particles, Particle* out_swarm) {
    // Basic Normal sampler (Box-Muller)
    double u1, u2;
    for(int i=0; i<n_particles; i++) {
        // Sample from the Gravity Distribution (Multivariate Normal)
        // Simplified: Sample independently from mean + small random noise
        u1 = (double)rand()/RAND_MAX; u2 = (double)rand()/RAND_MAX;
        double z = sqrt(-2.0*log(u1)) * cos(2.0*M_PI*u2);
        
        out_swarm[i].kappa = anchor->mean[0] + z * 0.1; // 0.1 is placeholder for sqrt(cov)
        out_swarm[i].theta = anchor->mean[1] + z * 0.01;
        // ... sample others ...
        out_swarm[i].weight = 1.0 / n_particles;
    }
}

// 2. The Filter Step (The Heart of the System)
void run_particle_filter_step(Particle* current_swarm, int n, double ret, double vol, double avg_vol, double dt, GravityDistribution* anchor, Particle* next_swarm, InstantState* out_state) {
    
    double total_weight = 0;
    
    // 1. Predict & Weight
    for(int i=0; i<n; i++) {
        Particle p = current_swarm[i];
        
        // Volume Clock
        double dt_eff = dt * (vol / avg_vol);
        
        // Predict expected return and variance
        double expected_ret = (p.mu - 0.5*p.theta)*dt_eff;
        double expected_var = p.theta*dt_eff + p.lambda_j*(p.mu_j*p.mu_j + p.sigma_j*p.sigma_j);
        
        // Calculate Likelihood (Fitness)
        double diff = ret - expected_ret;
        double likelihood = (1.0/sqrt(2*M_PI*expected_var)) * exp(-0.5*diff*diff/expected_var);
        
        p.weight = likelihood;
        total_weight += p.weight;
        
        current_swarm[i] = p; // Update weight in current swarm
    }
    
    // Normalize weights
    for(int i=0; i<n; i++) current_swarm[i].weight /= total_weight;
    
    // 2. Resample (Survival of the Fittest)
    // Systematic Resampling
    int idx = rand() % n;
    double beta = 0.0;
    double max_w = 0;
    for(int i=0; i<n; i++) if(current_swarm[i].weight > max_w) max_w = current_swarm[i].weight;
    
    for(int i=0; i<n; i++) {
        beta += (double)rand()/RAND_MAX * 2.0 * max_w;
        while(beta > current_swarm[idx].weight) {
            beta -= current_swarm[idx].weight;
            idx = (idx + 1) % n;
        }
        next_swarm[i] = current_swarm[idx];
    }
    
    // 3. Mutate (Evolve)
    for(int i=0; i<n; i++) {
        // Add small random noise to prevent particle collapse
        next_swarm[i].theta *= (1.0 + 0.01*((double)rand()/RAND_MAX - 0.5));
        // ... mutate others ...
    }
    
    // 4. Calculate Output Metrics
    // Expected Return = Weighted Mean of Mu
    double exp_ret = 0, exp_vol = 0, entropy = 0;
    for(int i=0; i<n; i++) {
        exp_ret += next_swarm[i].weight * next_swarm[i].mu;
        exp_vol += next_swarm[i].weight * next_swarm[i].theta;
        if(next_swarm[i].weight > 1e-9) entropy -= next_swarm[i].weight * log(next_swarm[i].weight);
    }
    
    out_state->expected_return = exp_ret;
    out_state->expected_vol = sqrt(exp_vol);
    out_state->swarm_entropy = entropy;
    
    // Mahalanobis & KL Divergence (Placeholder for complex math)
    out_state->mahalanobis_dist = fabs(exp_vol - sqrt(anchor->mean[1]));
    out_state->kl_divergence = 0.0; // Requires full dist comparison
}