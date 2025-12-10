#include "svcj.h"
#include <float.h>

// --- Helpers & Statistics ---
int compare_doubles(const void* a, const void* b) {
    double arg1=*(double*)a; double arg2=*(double*)b;
    return (arg1 > arg2) - (arg1 < arg2);
}

void sort_doubles_fast(double* arr, int n) {
    qsort(arr, n, sizeof(double), compare_doubles);
}

// Normal CDF for P-Values
double norm_cdf(double x) {
    return 0.5 * erfc(-x * 0.70710678);
}

// KS-Test to compare distributions
void perform_ks_test(Swarm* s1, Swarm* s2, double* out_stat, double* out_p) {
    int n1 = s1->n_particles;
    int n2 = s2->n_particles;
    
    // Extract Thetas
    double* t1 = malloc(n1 * sizeof(double));
    double* t2 = malloc(n2 * sizeof(double));
    for(int i=0; i<n1; i++) t1[i] = s1->particles[i].theta;
    for(int i=0; i<n2; i++) t2[i] = s2->particles[i].theta;
    
    sort_doubles_fast(t1, n1);
    sort_doubles_fast(t2, n2);
    
    double d_max = 0;
    int i=0, j=0;
    while(i<n1 && j<n2) {
        double cdf1 = (double)i/n1; double cdf2 = (double)j/n2;
        double diff = fabs(cdf1-cdf2);
        if(diff > d_max) d_max = diff;
        if(t1[i] < t2[j]) i++; else j++;
    }
    
    *out_stat = d_max;
    // KS P-Value Approximation
    double ks_lambda = (sqrt((double)n1*n2/(n1+n2)) + 0.12 + 0.11/sqrt((double)n1*n2/(n1+n2))) * d_max;
    *out_p = 2.0 * exp(-2.0 * ks_lambda * ks_lambda);
    
    free(t1); free(t2);
}

// --- Data Prep ---
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) {
    for(int i=1; i<n_rows; i++) {
        double prev=ohlcv[(i-1)*N_COLS+3]; double curr=ohlcv[i*N_COLS+3];
        if(prev<1e-9) prev=1e-9;
        out_returns[i-1] = log(curr/prev);
    }
}

// --- Particle Filter Core ---

// 1. Seeding the Swarm
// We use a simplified optimizer to find the center of mass for the Prior
void generate_prior_swarm(double* ohlcv, int n, double dt, Swarm* out, int n_particles) {
    // 1a. Simplified Optimizer to get Anchor Point
    SVCJParams p;
    // ... Placeholder for optimize_svcj logic (Nelder-Mead). 
    // This finds the Maximum Likelihood Estimate (MLE)
    double sum_sq = 0;
    double* ret = malloc((n-1)*sizeof(double));
    compute_log_returns(ohlcv, n, ret);
    for(int i=0; i<n-1; i++) sum_sq += ret[i]*ret[i];
    p.theta = (sum_sq/(n-1))/dt;
    p.kappa = 3.0; p.sigma_v = 0.5; p.rho = -0.5; p.lambda_j = 0.5;
    
    // 1b. Particle Nursery (Sample around the Anchor)
    out->particles = malloc(n_particles * sizeof(Particle));
    out->n_particles = n_particles;
    
    for(int i=0; i<n_particles; i++) {
        // Sample from Normal dist (Box-Muller)
        double u1 = (double)rand()/RAND_MAX;
        double u2 = (double)rand()/RAND_MAX;
        double z = sqrt(-2.0*log(u1))*cos(2.0*M_PI*u2);
        
        out->particles[i].theta = p.theta + z * p.theta * 0.2; // 20% std dev
        if(out->particles[i].theta < 1e-6) out->particles[i].theta = 1e-6;
        
        out->particles[i].kappa = p.kappa + z * 0.5;
        if(out->particles[i].kappa < 0.1) out->particles[i].kappa = 0.1;
        
        out->particles[i].sigma_v = p.sigma_v;
        out->particles[i].rho = p.rho;
        out->particles[i].lambda_j = p.lambda_j;
        
        out->particles[i].v_state = out->particles[i].theta;
        out->particles[i].weight = 1.0 / n_particles;
    }
    free(ret);
}

// 2. Evolving the Swarm
// This function runs the core Fork-Weight-Resample loop
void evolve_swarm(Swarm* current, double* returns, int n_ret, double dt, Swarm* next) {
    int N = current->n_particles;
    double sum_weights = 0;
    
    // For each particle...
    for(int i=0; i<N; i++) {
        Particle p = current->particles[i];
        
        // A. Prediction (Forking/Mutation)
        // Add random noise to parameters to allow exploration
        double z = sqrt(-2.0*log((double)rand()/RAND_MAX))*cos(2.0*M_PI*((double)rand()/RAND_MAX));
        p.theta += z * 0.001;
        if(p.theta<1e-6)p.theta=1e-6;
        
        // B. Weighting (Bayesian Update)
        // How well does this particle explain the data?
        double ll = 0;
        double v = p.v_state;
        for(int t=0; t<n_ret; t++) {
            double v_pred = v + p.kappa*(p.theta-v)*dt;
            if(v_pred < 1e-7) v_pred = 1e-7;
            
            double y = returns[t]; // Simplified drift
            double S = v_pred*dt; // Simplified variance
            if(S<1e-9)S=1e-9;
            
            double pdf = (1.0/sqrt(S*2*M_PI)) * exp(-0.5*y*y/S);
            ll += log(pdf + 1e-20);
            v = v_pred + 0.1*(y*y - S); // Simplified Update
        }
        
        p.weight = exp(ll);
        if(isnan(p.weight) || isinf(p.weight)) p.weight = 0;
        
        p.v_state = v; // Store final state
        
        current->particles[i] = p; // Update particle
        sum_weights += p.weight;
    }
    
    // Normalize weights
    for(int i=0; i<N; i++) {
        current->particles[i].weight /= sum_weights;
    }
    
    // C. Resampling (Survival of the Fittest)
    // Systematic Resampling for efficiency
    double* cdf = malloc(N * sizeof(double));
    cdf[0] = current->particles[0].weight;
    for(int i=1; i<N; i++) cdf[i] = cdf[i-1] + current->particles[i].weight;
    
    double u = (double)rand()/RAND_MAX / N;
    int j = 0;
    for(int i=0; i<N; i++) {
        while(u > cdf[j]) j++;
        next->particles[i] = current->particles[j];
        u += 1.0/N;
    }
    
    free(cdf);
    
    // Update Swarm Stats
    next->avg_theta = 0; next->avg_kappa = 0; next->avg_spot_vol = 0;
    for(int i=0; i<N; i++) {
        next->avg_theta += next->particles[i].theta;
        next->avg_kappa += next->particles[i].kappa;
        next->avg_spot_vol += sqrt(next->particles[i].v_state);
    }
    next->avg_theta/=N; next->avg_kappa/=N; next->avg_spot_vol/=N;
}

// --- THE SIGNAL ENGINE ---
void run_hierarchical_scan(double* ohlcv, int len_long, int len_short, double dt, int n_particles, BreakoutSignal* out) {
    if(len_long < len_short + 10) { out->is_breakout = 0; return; }
    
    // 1. Gravity Swarm (SLOW)
    Swarm gravity;
    generate_prior_swarm(ohlcv, len_long, dt, &gravity, n_particles);
    
    // 2. Impulse Swarm (FAST)
    Swarm impulse;
    int offset = len_long - len_short;
    generate_prior_swarm(ohlcv + offset*N_COLS, len_short, dt, &impulse, n_particles);
    
    // 3. Statistical Test: Compare the Swarms
    perform_ks_test(&gravity, &impulse, &out->ks_stat, &out->p_value);
    
    // 4. Calculate Physics
    out->energy_ratio = impulse.avg_spot_vol / sqrt(gravity.avg_theta);
    
    double res_sum = 0;
    double* ret_short = malloc((len_short-1)*sizeof(double));
    compute_log_returns(ohlcv + offset*N_COLS, len_short, ret_short);
    for(int i=0; i<len_short-1; i++) res_sum += ret_short[i];
    out->drift_z_score = res_sum; // Simplified
    
    // 5. Decision
    // Breakout is REAL if:
    // a) The distributions are statistically different (p < 0.05)
    // b) The energy expanded
    // c) The drift is positive
    
    if(out->p_value < 0.05 && out->energy_ratio > 1.2 && out->drift_z_score > 0) {
        out->is_breakout = 1;
    } else {
        out->is_breakout = 0;
    }
    
    free(gravity.particles);
    free(impulse.particles);
    free(ret_short);
}