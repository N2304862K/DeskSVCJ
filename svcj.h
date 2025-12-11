#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define N_STATES 3 // Bull, Bear, Neutral
#define MAX_ITER 50 // Baum-Welch Iterations
#define SQRT_2PI 2.50662827463
#define N_COLS 5

// The Physics of a SINGLE Regime (State)
typedef struct {
    double mu;           // Drift
    double sigma;        // Volatility
} HMMStateParams;

// The Full HMM Model
typedef struct {
    HMMStateParams states[N_STATES];
    double transitions[N_STATES][N_STATES]; // A[i][j] = Prob(State i -> State j)
    double initial_probs[N_STATES];
} HMMModel;

// Core Utils
void compute_log_returns(double* ohlcv, int n, double* out_returns);

// Baum-Welch Algorithm (The "Solver")
void run_baum_welch(double* returns, int n, HMMModel* model);

// Viterbi Algorithm (The "Decoder")
void decode_states_viterbi(double* returns, int n, HMMModel* model, int* out_path);

#endif