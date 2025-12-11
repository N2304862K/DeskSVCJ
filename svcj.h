#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define N_STATES 3 // 0=Bull, 1=Bear, 2=Neutral
#define N_COLS 5
#define SQRT_2PI 2.50662827463

// The Physics of a SINGLE Regime
typedef struct {
    double mu;     // Drift
    double sigma;  // Volatility (Simplified from full SVCJ for HMM state)
} RegimeParams;

// The Complete HMM Model
typedef struct {
    RegimeParams states[N_STATES];
    double transitions[N_STATES][N_STATES]; // A[i][j] = Prob of going from State i to j
    double initial_probs[N_STATES];       // Pi
} HMM;

// Core Utils
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns);

// Main HMM Engine
void train_svcj_hmm(double* returns, int n, double dt, int max_iter, HMM* out_model);

// Viterbi Algorithm (Finds the most likely path of hidden states)
void decode_regime_path(double* returns, int n, double dt, HMM* model, int* out_path);

#endif