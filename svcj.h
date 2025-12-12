#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define N_COLS 5
#define N_REGIMES 3 // Bull, Bear, Crash

typedef struct {
    double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j;
} SVCJParams;

typedef struct {
    // Current State
    double probabilities[N_REGIMES]; // P(Regime_i)
    int most_likely_regime;          // Viterbi path output
    
    // Weighted Physics (For external use)
    double expected_spot_vol;
    double expected_lambda;
} HMMState;

// Core Utils
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

// HMM Engine
void run_hmm_forward_pass(
    double return_val, 
    double dt,
    SVCJParams* params_array,         // Array of SVCJParams for each regime
    double* transition_matrix,        // N_REGIMES x N_REGIMES
    double* last_probs,               // Input Probabilities
    double* out_probs,                // Output Probabilities
    double* out_likelihoods          // Likelihood of data given each regime
);

void viterbi_decode(
    int n_obs,
    double* all_likelihoods,          // T x N_REGIMES matrix
    double* transition_matrix,
    double* initial_probs,
    int* out_path                    // Most likely sequence of regimes
);

// Helper for pure likelihood calc
double ukf_single_pass_likelihood(
    double return_val, 
    double dt, 
    SVCJParams* p, 
    double* state_var
);

#endif