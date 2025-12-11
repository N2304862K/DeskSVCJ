#include "svcj.h"
#include <float.h>

void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) {
    for(int i=1; i<n_rows; i++) {
        double prev = ohlcv[(i-1)*N_COLS + 3];
        double curr = ohlcv[i*N_COLS + 3];
        if(prev < 1e-9) prev = 1e-9;
        out_returns[i-1] = log(curr / prev);
    }
}

// --- HMM ENGINE ---

// Gaussian PDF: Prob of observing return 'x' given a Regime's physics
double emission_prob(double x, double dt, RegimeParams* state) {
    double mean = state->mu * dt;
    double std = state->sigma * sqrt(dt);
    if(std < 1e-9) std = 1e-9;
    
    double exponent = -0.5 * pow((x - mean) / std, 2);
    return (1.0 / (SQRT_2PI * std)) * exp(exponent);
}

void train_svcj_hmm(double* returns, int n, double dt, int max_iter, HMM* model) {
    // 1. Initialization (K-Means or Random)
    // Random Init: Bull (High Drift), Bear (Low), Neut (Mid)
    model->states[0] = (RegimeParams){0.20, 0.40};  // Bull
    model->states[1] = (RegimeParams){-0.20, 0.40}; // Bear
    model->states[2] = (RegimeParams){0.0, 0.15};   // Neutral
    
    // Uniform transitions
    for(int i=0; i<N_STATES; i++) {
        model->initial_probs[i] = 1.0/N_STATES;
        for(int j=0; j<N_STATES; j++) {
            model->transitions[i][j] = 1.0/N_STATES;
        }
    }
    
    // Buffers for Baum-Welch
    double* alpha = malloc(n * N_STATES * sizeof(double)); // Forward
    double* beta = malloc(n * N_STATES * sizeof(double));  // Backward
    double* c = malloc(n * sizeof(double));              // Scale factors
    double** gamma = malloc(n * sizeof(double*));
    double*** xi = malloc(n * sizeof(double**));
    for(int i=0; i<n; i++) {
        gamma[i] = malloc(N_STATES * sizeof(double));
        xi[i] = malloc(N_STATES * sizeof(double*));
        for(int j=0; j<N_STATES; j++) xi[i][j] = malloc(N_STATES * sizeof(double));
    }

    // --- BAUM-WELCH (EM) LOOP ---
    for(int iter=0; iter<max_iter; iter++) {
        // --- E-STEP (Expectation) ---
        
        // 1. Forward Pass (Alpha)
        c[0] = 0;
        for(int i=0; i<N_STATES; i++) {
            alpha[0*N_STATES+i] = model->initial_probs[i] * emission_prob(returns[0], dt, &model->states[i]);
            c[0] += alpha[0*N_STATES+i];
        }
        c[0] = 1.0 / c[0];
        for(int i=0; i<N_STATES; i++) alpha[0*N_STATES+i] *= c[0];
        
        for(int t=1; t<n; t++) {
            c[t] = 0;
            for(int i=0; i<N_STATES; i++) {
                alpha[t*N_STATES+i] = 0;
                for(int j=0; j<N_STATES; j++) {
                    alpha[t*N_STATES+i] += alpha[(t-1)*N_STATES+j] * model->transitions[j][i];
                }
                alpha[t*N_STATES+i] *= emission_prob(returns[t], dt, &model->states[i]);
                c[t] += alpha[t*N_STATES+i];
            }
            c[t] = 1.0 / c[t];
            for(int i=0; i<N_STATES; i++) alpha[t*N_STATES+i] *= c[t];
        }
        
        // 2. Backward Pass (Beta)
        for(int i=0; i<N_STATES; i++) beta[(n-1)*N_STATES+i] = c[n-1];
        
        for(int t=n-2; t>=0; t--) {
            for(int i=0; i<N_STATES; i++) {
                beta[t*N_STATES+i] = 0;
                for(int j=0; j<N_STATES; j++) {
                    beta[t*N_STATES+i] += model->transitions[i][j] * emission_prob(returns[t+1], dt, &model->states[j]) * beta[(t+1)*N_STATES+j];
                }
                beta[t*N_STATES+i] *= c[t];
            }
        }
        
        // 3. Gamma & Xi (Responsibilities)
        double ll = 0;
        for(int i=0; i<n; i++) ll -= log(c[i]);
        
        for(int t=0; t<n-1; t++) {
            double den = 0;
            for(int i=0; i<N_STATES; i++) {
                for(int j=0; j<N_STATES; j++) {
                    den += alpha[t*N_STATES+i] * model->transitions[i][j] * emission_prob(returns[t+1], dt, &model->states[j]) * beta[(t+1)*N_STATES+j];
                }
            }
            for(int i=0; i<N_STATES; i++) {
                gamma[t][i] = 0;
                for(int j=0; j<N_STATES; j++) {
                    xi[t][i][j] = (alpha[t*N_STATES+i] * model->transitions[i][j] * emission_prob(returns[t+1], dt, &model->states[j]) * beta[(t+1)*N_STATES+j]) / den;
                    gamma[t][i] += xi[t][i][j];
                }
            }
        }
        // Handle last time step for gamma
        double den_last = 0;
        for(int i=0; i<N_STATES; i++) den_last += alpha[(n-1)*N_STATES+i];
        for(int i=0; i<N_STATES; i++) gamma[n-1][i] = alpha[(n-1)*N_STATES+i] / den_last;
        
        // --- M-STEP (Maximization) ---
        
        // 1. Re-estimate Initial Probs
        for(int i=0; i<N_STATES; i++) model->initial_probs[i] = gamma[0][i];
        
        // 2. Re-estimate Transitions
        for(int i=0; i<N_STATES; i++) {
            double gamma_sum = 0;
            for(int t=0; t<n-1; t++) gamma_sum += gamma[t][i];
            
            for(int j=0; j<N_STATES; j++) {
                double xi_sum = 0;
                for(int t=0; t<n-1; t++) xi_sum += xi[t][i][j];
                model->transitions[i][j] = xi_sum / gamma_sum;
            }
        }
        
        // 3. Re-estimate State Physics (Mu, Sigma)
        for(int i=0; i<N_STATES; i++) {
            double gamma_sum_total = 0;
            double num_mu = 0, num_sig = 0;
            
            for(int t=0; t<n; t++) {
                gamma_sum_total += gamma[t][i];
                num_mu += gamma[t][i] * returns[t];
            }
            
            double new_mu = (num_mu / gamma_sum_total) / dt; // Annualize
            
            for(int t=0; t<n; t++) {
                double diff = returns[t] - (new_mu * dt);
                num_sig += gamma[t][i] * diff * diff;
            }
            double new_sigma = sqrt((num_sig / gamma_sum_total) / dt);
            
            model->states[i].mu = new_mu;
            model->states[i].sigma = new_sigma;
        }
    }
    
    // Cleanup
    free(alpha); free(beta); free(c);
    for(int i=0; i<n; i++) {
        free(gamma[i]);
        for(int j=0; j<N_STATES; j++) free(xi[i][j]);
        free(xi[i]);
    }
    free(gamma); free(xi);
}

// --- Viterbi Decoder ---
void decode_regime_path(double* returns, int n, double dt, HMM* model, int* out_path) {
    double* T1 = malloc(n * N_STATES * sizeof(double)); // Probabilities
    int* T2 = malloc(n * N_STATES * sizeof(int));    // Pointers

    // Init
    for(int i=0; i<N_STATES; i++) {
        T1[0*N_STATES+i] = log(model->initial_probs[i]) + log(emission_prob(returns[0], dt, &model->states[i]));
        T2[0*N_STATES+i] = 0;
    }

    // Forward
    for(int t=1; t<n; t++) {
        for(int j=0; j<N_STATES; j++) {
            double max_prob = -DBL_MAX;
            int max_idx = 0;
            for(int i=0; i<N_STATES; i++) {
                double prob = T1[(t-1)*N_STATES+i] + log(model->transitions[i][j]);
                if (prob > max_prob) {
                    max_prob = prob;
                    max_idx = i;
                }
            }
            T1[t*N_STATES+j] = max_prob + log(emission_prob(returns[t], dt, &model->states[j]));
            T2[t*N_STATES+j] = max_idx;
        }
    }
    
    // Backtrack
    double final_max = -DBL_MAX;
    for(int i=0; i<N_STATES; i++) {
        if(T1[(n-1)*N_STATES+i] > final_max) {
            final_max = T1[(n-1)*N_STATES+i];
            out_path[n-1] = i;
        }
    }

    for(int t=n-2; t>=0; t--) {
        out_path[t] = T2[(t+1)*N_STATES + out_path[t+1]];
    }
    
    free(T1); free(T2);
}