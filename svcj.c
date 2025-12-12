#include "svcj.h"
#include <float.h>

// --- Utils ---
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) {
    for(int i=1; i<n_rows; i++) {
        double p1 = ohlcv[(i-1)*N_COLS + 3];
        double p2 = ohlcv[i*N_COLS + 3];
        if(p1 > 1e-9) out_returns[i-1] = log(p2 / p1);
        else out_returns[i-1] = 0.0;
    }
}

// --- HMM Core ---
// Calculates Likelihood(Return | Params) for a SINGLE regime
double ukf_single_pass_likelihood(double r, double dt, SVCJParams* p, double* v_state) {
    // 1. Predict variance
    double v_pred = *v_state + p->kappa * (p->theta - *v_state) * dt;
    if(v_pred < 1e-9) v_pred = 1e-9;
    
    // 2. Innovation
    double y = r - (p->mu - 0.5 * v_pred) * dt;
    
    // 3. Expected Variance (Diffusive + Jump)
    double var_j = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
    double total_var = (v_pred + var_j) * dt;
    if(total_var < 1e-12) total_var = 1e-12;
    
    // 4. Update State (for next iteration)
    double S = v_pred*dt + var_j*dt;
    double K = (p->rho * p->sigma_v * dt) / S;
    double v_new = v_pred + K * y;
    if(v_new < 1e-9) v_new = 1e-9;
    *v_state = v_new;
    
    // 5. Return PDF value (Likelihood)
    return (1.0 / sqrt(total_var * 2 * M_PI)) * exp(-0.5 * y*y / total_var);
}

// The Forward Algorithm Step
void run_hmm_forward_pass(
    double return_val, double dt,
    SVCJParams* params_array,
    double* trans_mat,
    double* last_probs,
    double* out_probs,
    double* out_likelihoods)
{
    // 1. Prediction Step (alpha_t|t-1)
    // Sum over previous states: P(S_t=j) = Sum_i P(S_t=j|S_{t-1}=i) * P(S_{t-1}=i)
    double pred_probs[N_REGIMES] = {0};
    for(int j=0; j<N_REGIMES; j++) {
        for(int i=0; i<N_REGIMES; i++) {
            // trans_mat[i*N_REGIMES + j] is P(j | i)
            pred_probs[j] += trans_mat[i*N_REGIMES + j] * last_probs[i];
        }
    }

    // 2. Observation Step (Calculate Likelihoods)
    // We need to maintain a variance state for each filter
    // For this pass, we can simplify by assuming it starts at theta each time,
    // or pass a state array. Let's assume a simplified state.
    static double filter_states[N_REGIMES] = {0.04, 0.09, 0.25}; // Init
    
    for(int i=0; i<N_REGIMES; i++) {
        if(filter_states[i] == 0) filter_states[i] = params_array[i].theta;
        out_likelihoods[i] = ukf_single_pass_likelihood(return_val, dt, &params_array[i], &filter_states[i]);
    }
    
    // 3. Update Step (alpha_t)
    // NewProb = PredictedProb * Likelihood
    double total_prob = 0;
    for(int i=0; i<N_REGIMES; i++) {
        out_probs[i] = pred_probs[i] * out_likelihoods[i];
        total_prob += out_probs[i];
    }
    
    // Normalize
    if (total_prob > 1e-12) {
        for(int i=0; i<N_REGIMES; i++) {
            out_probs[i] /= total_prob;
        }
    }
}

// Viterbi Path (Noise-Resistant Regime)
void viterbi_decode(
    int n_obs,
    double* all_likelihoods, // T x N_REGIMES
    double* trans_mat,
    double* initial_probs,
    int* out_path)
{
    double T1[N_REGIMES];
    int T2[n_obs][N_REGIMES];
    
    // Init
    for(int i=0; i<N_REGIMES; i++) {
        T1[i] = log(initial_probs[i]) + log(all_likelihoods[i]); // Log space
    }
    
    // Forward Pass
    for(int t=1; t<n_obs; t++) {
        double T1_new[N_REGIMES];
        for(int j=0; j<N_REGIMES; j++) {
            double max_prob = -DBL_MAX;
            int max_idx = 0;
            for(int i=0; i<N_REGIMES; i++) {
                double p = T1[i] + log(trans_mat[i*N_REGIMES + j]);
                if (p > max_prob) {
                    max_prob = p;
                    max_idx = i;
                }
            }
            T1_new[j] = max_prob + log(all_likelihoods[t*N_REGIMES + j]);
            T2[t][j] = max_idx;
        }
        memcpy(T1, T1_new, N_REGIMES * sizeof(double));
    }
    
    // Backtrack
    double max_final = -DBL_MAX;
    int last_state = 0;
    for(int i=0; i<N_REGIMES; i++) {
        if(T1[i] > max_final) {
            max_final = T1[i];
            last_state = i;
        }
    }
    
    out_path[n_obs-1] = last_state;
    for(int t=n_obs-2; t>=0; t--) {
        out_path[t] = T2[t+1][out_path[t+1]];
    }
}