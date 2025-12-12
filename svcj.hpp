#ifndef SVCJ_HPP
#define SVCJ_HPP

#include <cmath>
#include <vector>

#define N_REGIMES 3
#define N_PARTICLES 500 // Number of particles for the filter

// C-style compatibility for Cython
#ifdef __cplusplus
extern "C" {
#endif

// --- Data Structures ---

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

// A single particle in the filter
typedef struct {
    double v; // The state (variance)
    double w; // The weight
} Particle;

// The full HMM state output
typedef struct {
    double probabilities[N_REGIMES];
    double spot_vol_dist[10]; // A 10-bin histogram of the spot vol distribution
    int most_likely_regime;
} HMMState;


// --- Function Prototypes ---

// LUT
void init_lut();
double fast_exp(double x);

// Core HMM Logic
void run_hmm_forward_pass_cpp(
    double return_val, 
    double dt,
    SVCJParams* params_array,
    double* transition_matrix,
    double* last_probs,
    Particle** particle_clouds, // Array of particle clouds
    HMMState* out_state
);

#ifdef __cplusplus
}
#endif

#endif // SVCJ_HPP