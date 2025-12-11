#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MAX_STATES 5
#define MAX_ITER 100
#define N_COLS 5

// --- Structures ---

// Holds the parameters of a trained HMM
typedef struct {
    int n_states;
    double initial_probs[MAX_STATES];
    double transitions[MAX_STATES][MAX_STATES];
    // Emission parameters (Gaussian: mean, variance)
    double means[MAX_STATES];
    double variances[MAX_STATES];
} HMMModel;

// The final output of the training process
typedef struct {
    HMMModel model;
    double log_likelihood;
    // The Viterbi Path (most likely sequence of states)
    int* viterbi_path;
} HMMResult;

// --- Function Prototypes ---

// Main entry point for training
void train_hmm(double* ohlcv, int n_obs, int n_states, double dt, HMMResult* result);

#endif