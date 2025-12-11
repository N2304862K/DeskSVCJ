#include "svcj.h"
#include <float.h>

// --- Utils ---
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) {
    for(int i=1; i<n_rows; i++) {
        double prev = ohlcv[(i-1)*N_COLS + 3];
        double curr = ohlcv[i*N_COLS + 3];
        if(prev < 1e-9) prev = 1e-9;
        out_returns[i-1] = log(curr / prev);
    }
}

// Gaussian PDF
double gaussian_prob(double x, double mu, double sigma) {
    if (sigma < 1e-9) sigma = 1e-9;
    double exponent = -0.5 * pow((x - mu) / sigma, 2);
    return (1.0 / (SQRT_2PI * sigma)) * exp(exponent);
}

// --- Baum-Welch (The Solver) ---
void run_baum_welch(double* returns, int n, HMMModel* model) {
    // 1. Initialization (Reasonable Guesses)
    // State 0: Bull, State 1: Bear, State 2: Neutral
    model->states[0].mu = 0.0005; model->states[0].sigma = 0.005;
    model->states[1].mu = -0.0005; model->states[1].sigma = 0.01;
    model->states[2].mu = 0.0; model->states[2].sigma = 0.002;
    
    // Transition Matrix: High prob of staying in same state
    for(int i=0; i<N_STATES; i++) {
        for(int j=0; j<N_STATES; j++) {
            model->transitions[i][j] = (i==j) ? 0.98 : 0.01;
        }
    }
    
    model->initial_probs[0] = 0.33; model->initial_probs[1] = 0.33; model->initial_probs[2] = 0.34;

    // Allocate Memory for Forward/Backward Pass
    double* alpha = malloc(n * N_STATES * sizeof(double));
    double* beta = malloc(n * N_STATES * sizeof(double));
    double* c = malloc(n * sizeof(double)); // Scaling factor

    // EM Loop
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // --- E-Step: Forward-Backward Pass ---
        
        // Forward (alpha)
        c[0] = 0;
        for(int i=0; i<N_STATES; i++) {
            alpha[i] = model->initial_probs[i] * gaussian_prob(returns[0], model->states[i].mu, model->states[i].sigma);
            c[0] += alpha[i];
        }
        if(c[0] > 1e-9) for(int i=0; i<N_STATES; i++) alpha[i] /= c[0]; // Scale
        
        for(int t=1; t<n; t++) {
            c[t] = 0;
            for(int i=0; i<N_STATES; i++) {
                double alpha_sum = 0;
                for(int j=0; j<N_STATES; j++) {
                    alpha_sum += alpha[(t-1)*N_STATES + j] * model->transitions[j][i];
                }
                alpha[t*N_STATES + i] = alpha_sum * gaussian_prob(returns[t], model->states[i].mu, model->states[i].sigma);
                c[t] += alpha[t*N_STATES + i];
            }
            if(c[t] > 1e-9) for(int i=0; i<N_STATES; i++) alpha[t*N_STATES + i] /= c[t]; // Scale
        }
        
        // Backward (beta)
        for(int i=0; i<N_STATES; i++) beta[(n-1)*N_STATES + i] = 1.0;
        
        for(int t=n-2; t>=0; t--) {
            for(int i=0; i<N_STATES; i++) {
                beta[t*N_STATES + i] = 0;
                for(int j=0; j<N_STATES; j++) {
                    beta[t*N_STATES + i] += model->transitions[i][j] * gaussian_prob(returns[t+1], model->states[j].mu, model->states[j].sigma) * beta[(t+1)*N_STATES + j];
                }
                beta[t*N_STATES + i] /= c[t+1]; // Scale
            }
        }
        
        // --- M-Step: Re-estimate Parameters ---
        double gamma[N_STATES], xi[N_STATES][N_STATES];
        
        // Transitions
        for(int i=0; i<N_STATES; i++) {
            double den = 0;
            for(int t=0; t<n-1; t++) {
                den += alpha[t*N_STATES + i] * beta[t*N_STATES + i];
            }
            if(den < 1e-9) den = 1e-9;
            
            for(int j=0; j<N_STATES; j++) {
                double num = 0;
                for(int t=0; t<n-1; t++) {
                    num += alpha[t*N_STATES + i] * model->transitions[i][j] * gaussian_prob(returns[t+1], model->states[j].mu, model->states[j].sigma) * beta[(t+1)*N_STATES + j];
                }
                model->transitions[i][j] = num / den;
            }
        }
        
        // Emissions (Mu, Sigma)
        for(int i=0; i<N_STATES; i++) {
            double gamma_sum = 0;
            double mu_num = 0, sigma_num = 0;
            
            for(int t=0; t<n; t++) {
                double g = alpha[t*N_STATES + i] * beta[t*N_STATES + i];
                gamma_sum += g;
                mu_num += g * returns[t];
            }
            if(gamma_sum < 1e-9) gamma_sum = 1e-9;
            double new_mu = mu_num / gamma_sum;
            model->states[i].mu = new_mu;
            
            for(int t=0; t<n; t++) {
                double g = alpha[t*N_STATES + i] * beta[t*N_STATES + i];
                sigma_num += g * pow(returns[t] - new_mu, 2);
            }
            model->states[i].sigma = sqrt(sigma_num / gamma_sum);
        }
    }
    free(alpha); free(beta); free(c);
}

// --- Viterbi (The Decoder) ---
void decode_states_viterbi(double* returns, int n, HMMModel* model, int* out_path) {
    double* T1 = malloc(n * N_STATES * sizeof(double));
    int* T2 = malloc(n * N_STATES * sizeof(int));
    
    // Init
    for(int i=0; i<N_STATES; i++) {
        T1[i] = log(model->initial_probs[i] + 1e-30) + log(gaussian_prob(returns[0], model->states[i].mu, model->states[i].sigma) + 1e-30);
    }
    
    // Forward
    for(int t=1; t<n; t++) {
        for(int i=0; i<N_STATES; i++) {
            double max_prob = -1e30;
            int max_idx = 0;
            for(int j=0; j<N_STATES; j++) {
                double prob = T1[(t-1)*N_STATES + j] + log(model->transitions[j][i] + 1e-30);
                if (prob > max_prob) {
                    max_prob = prob;
                    max_idx = j;
                }
            }
            T1[t*N_STATES + i] = max_prob + log(gaussian_prob(returns[t], model->states[i].mu, model->states[i].sigma) + 1e-30);
            T2[t*N_STATES + i] = max_idx;
        }
    }
    
    // Find last state
    double max_final = -1e30;
    int last_state = 0;
    for(int i=0; i<N_STATES; i++) {
        if(T1[(n-1)*N_STATES + i] > max_final) {
            max_final = T1[(n-1)*N_COLS + i];
            last_state = i;
        }
    }
    out_path[n-1] = last_state;
    
    // Backtrack
    for(int t=n-2; t>=0; t--) {
        out_path[t] = T2[(t+1)*N_STATES + out_path[t+1]];
    }
    
    free(T1); free(T2);
}