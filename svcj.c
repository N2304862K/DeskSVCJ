#include "svcj.h"
#include <float.h>

// --- Helper Functions ---
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) {
    for(int i=1; i<n_rows; i++) {
        double prev = ohlcv[(i-1)*N_COLS + 3];
        double curr = ohlcv[i*N_COLS + 3];
        if(prev < 1e-9) prev = 1e-9;
        out_returns[i-1] = log(curr / prev);
    }
}

double gaussian_pdf(double x, double mean, double var) {
    if (var < 1e-12) var = 1e-12;
    double coeff = 1.0 / sqrt(2.0 * M_PI * var);
    double exponent = -0.5 * (x - mean) * (x - mean) / var;
    return coeff * exp(exponent);
}

// --- HMM Core Algorithms ---
void train_hmm(double* ohlcv, int n_obs, int n_states, double dt, HMMResult* result) {
    int n_ret = n_obs - 1;
    double* returns = malloc(n_ret * sizeof(double));
    compute_log_returns(ohlcv, n_obs, returns);
    
    // 1. Initialization
    HMMModel model;
    model.n_states = n_states;
    
    for(int i=0; i<n_states; i++) {
        model.initial_probs[i] = 1.0 / n_states;
        for(int j=0; j<n_states; j++) {
            model.transitions[i][j] = 1.0 / n_states;
        }
    }
    
    int chunk_size = n_ret / n_states;
    for(int i=0; i<n_states; i++) {
        double sum = 0, sum_sq = 0;
        int start = i * chunk_size;
        int end = (i + 1) * chunk_size;
        for(int t=start; t<end; t++) {
            sum += returns[t];
            sum_sq += returns[t] * returns[t];
        }
        model.means[i] = sum / chunk_size;
        model.variances[i] = (sum_sq / chunk_size) - (model.means[i] * model.means[i]);
    }
    
    double* alpha = malloc(n_ret * n_states * sizeof(double));
    double* beta = malloc(n_ret * n_states * sizeof(double));
    double* gamma = malloc(n_ret * n_states * sizeof(double));
    double* xi = malloc(n_ret * n_states * n_states * sizeof(double));
    double* scale = malloc(n_ret * sizeof(double));
    
    double old_log_lik = -DBL_MAX;
    
    for(int iter=0; iter<MAX_ITER; iter++) {
        // --- E-Step ---
        // Forward
        scale[0] = 0.0;
        for(int i=0; i<n_states; i++) {
            alpha[i] = model.initial_probs[i] * gaussian_pdf(returns[0], model.means[i], model.variances[i]);
            scale[0] += alpha[i];
        }
        for(int i=0; i<n_states; i++) alpha[i] /= scale[0];
        
        for(int t=1; t<n_ret; t++) {
            scale[t] = 0.0;
            for(int i=0; i<n_states; i++) {
                double sum = 0.0;
                for(int j=0; j<n_states; j++) {
                    sum += alpha[(t-1)*n_states + j] * model.transitions[j][i];
                }
                alpha[t*n_states + i] = sum * gaussian_pdf(returns[t], model.means[i], model.variances[i]);
                scale[t] += alpha[t*n_states + i];
            }
            if (scale[t] > 0) {
                for(int i=0; i<n_states; i++) alpha[t*n_states + i] /= scale[t];
            }
        }
        
        // Backward
        for(int i=0; i<n_states; i++) beta[(n_ret-1)*n_states + i] = 1.0;
        
        for(int t=n_ret-2; t>=0; t--) {
            for(int i=0; i<n_states; i++) {
                double sum = 0.0;
                for(int j=0; j<n_states; j++) {
                    sum += model.transitions[i][j] * gaussian_pdf(returns[t+1], model.means[j], model.variances[j]) * beta[(t+1)*n_states + j];
                }
                beta[t*n_states + i] = sum / scale[t+1];
            }
        }
        
        // Gamma and Xi
        for(int t=0; t<n_ret-1; t++) {
            double den = 0.0;
            for(int i=0; i<n_states; i++) den += alpha[t*n_states+i]*beta[t*n_states+i];
            
            for(int i=0; i<n_states; i++) {
                gamma[t*n_states+i] = (alpha[t*n_states+i]*beta[t*n_states+i]) / den;
                for(int j=0; j<n_states; j++) {
                    double num = alpha[t*n_states+i] * model.transitions[i][j] * gaussian_pdf(returns[t+1], model.means[j], model.variances[j]) * beta[(t+1)*n_states+j];
                    xi[t*n_states*n_states + i*n_states + j] = num / den / scale[t+1];
                }
            }
        }
        
        // --- M-Step ---
        for(int i=0; i<n_states; i++) model.initial_probs[i] = gamma[i];
        
        for(int i=0; i<n_states; i++) {
            double den = 0.0;
            for(int t=0; t<n_ret-1; t++) den += gamma[t*n_states+i];
            
            for(int j=0; j<n_states; j++) {
                double num = 0.0;
                for(int t=0; t<n_ret-1; t++) num += xi[t*n_states*n_states + i*n_states+j];
                model.transitions[i][j] = num / den;
            }
        }
        
        for(int i=0; i<n_states; i++) {
            double gamma_sum=0, mean_num=0, var_num=0;
            for(int t=0; t<n_ret; t++) {
                gamma_sum += gamma[t*n_states+i];
                mean_num += gamma[t*n_states+i] * returns[t];
            }
            model.means[i] = mean_num / gamma_sum;
            
            for(int t=0; t<n_ret; t++) {
                double diff = returns[t] - model.means[i];
                var_num += gamma[t*n_states+i] * diff * diff;
            }
            model.variances[i] = var_num / gamma_sum;
        }
        
        double log_lik = 0;
        for(int t=0; t<n_ret; t++) log_lik += log(scale[t]);
        
        if (fabs(log_lik - old_log_lik) < 0.001) break;
        old_log_lik = log_lik;
    }
    
    // Viterbi
    int* viterbi_path = malloc(n_ret * sizeof(int));
    double* T1 = malloc(n_ret * n_states * sizeof(double));
    int* T2 = malloc(n_ret * n_states * sizeof(int));
    
    for(int i=0; i<n_states; i++) {
        T1[i] = log(model.initial_probs[i]) + log(gaussian_pdf(returns[0], model.means[i], model.variances[i]));
    }
    
    for(int t=1; t<n_ret; t++) {
        for(int j=0; j<n_states; j++) {
            double max_val = -DBL_MAX;
            int max_idx = 0;
            for(int i=0; i<n_states; i++) {
                double val = T1[(t-1)*n_states+i] + log(model.transitions[i][j]);
                if (val > max_val) { max_val = val; max_idx = i; }
            }
            T1[t*n_states+j] = max_val + log(gaussian_pdf(returns[t], model.means[j], model.variances[j]));
            T2[t*n_states+j] = max_idx;
        }
    }
    
    // Backtrack
    double max_prob = -DBL_MAX;
    for(int i=0; i<n_states; i++) {
        if(T1[(n_ret-1)*n_states+i] > max_prob) { max_prob = T1[(n_ret-1)*n_states+i]; viterbi_path[n_ret-1] = i; }
    }
    for(int t=n_ret-2; t>=0; t--) {
        viterbi_path[t] = T2[(t+1)*n_states + viterbi_path[t+1]];
    }
    
    result->model = model;
    result->viterbi_path = viterbi_path;
    
    free(alpha); free(beta); free(gamma); free(xi); free(scale);
    free(T1); free(T2);
    free(returns);
}