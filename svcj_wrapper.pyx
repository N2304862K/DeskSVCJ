# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    
    void run_hmm_forward_pass(
        double r, double dt, SVCJParams* p_arr, double* t_mat, 
        double* last_p, double* out_p, double* out_l) nogil
    
    void viterbi_decode(
        int n, double* all_l, double* t_mat, double* init_p, int* path) nogil

# --- The HMM Engine Class ---
cdef class HMM_Engine:
    cdef SVCJParams* params_array
    cdef double* transition_matrix
    cdef double* current_probs
    cdef double dt
    cdef int n_regimes
    
    def __init__(self, list regime_params, np.ndarray[double, ndim=2, mode='c'] trans_mat, double dt):
        self.n_regimes = len(regime_params)
        self.dt = dt
        
        # Allocate C-Memory
        self.params_array = <SVCJParams*> malloc(self.n_regimes * sizeof(SVCJParams))
        self.transition_matrix = <double*> malloc(self.n_regimes * self.n_regimes * sizeof(double))
        self.current_probs = <double*> malloc(self.n_regimes * sizeof(double))
        
        # Populate Physics
        for i, p in enumerate(regime_params):
            self.params_array[i].mu = p.get('mu', 0.0)
            self.params_array[i].kappa = p['kappa']
            self.params_array[i].theta = p['theta']
            self.params_array[i].sigma_v = p['sigma_v']
            self.params_array[i].rho = p['rho']
            self.params_array[i].lambda_j = p['lambda_j']
            self.params_array[i].mu_j = p.get('mu_j', -0.05)
            self.params_array[i].sigma_j = p.get('sigma_j', 0.1)
        
        # Populate Transition Matrix
        for r in range(self.n_regimes):
            for c in range(self.n_regimes):
                self.transition_matrix[r * self.n_regimes + c] = trans_mat[r, c]
        
        # Initial State (Uniform Belief)
        for i in range(self.n_regimes):
            self.current_probs[i] = 1.0 / self.n_regimes
            
    def __dealloc__(self):
        free(self.params_array)
        free(self.transition_matrix)
        free(self.current_probs)

    def update(self, double ret):
        cdef np.ndarray[double, ndim=1] new_probs = np.zeros(self.n_regimes)
        cdef np.ndarray[double, ndim=1] likelihoods = np.zeros(self.n_regimes)
        
        run_hmm_forward_pass(
            ret, self.dt,
            self.params_array,
            self.transition_matrix,
            self.current_probs,
            <double*> new_probs.data,
            <double*> likelihoods.data
        )
        
        # Persist new state
        for i in range(self.n_regimes):
            self.current_probs[i] = new_probs[i]
            
        return new_probs, likelihoods

    @staticmethod
    def get_viterbi_path(np.ndarray[double, ndim=2, mode='c'] all_likelihoods,
                         np.ndarray[double, ndim=2, mode='c'] trans_mat,
                         np.ndarray[double, ndim=1, mode='c'] init_probs):
        cdef int n_obs = all_likelihoods.shape[0]
        cdef int n_regimes = all_likelihoods.shape[1]
        cdef np.ndarray[int, ndim=1, mode='c'] path = np.zeros(n_obs, dtype=np.int32)
        
        viterbi_decode(n_obs, <double*> all_likelihoods.data, <double*> trans_mat.data, 
                       <double*> init_probs.data, <int*> path.data)
        
        return path