# distutils: language = cpp

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.hpp":
    cdef int N_REGIMES
    cdef int N_PARTICLES
    
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct Particle:
        double v, w
    ctypedef struct HMMState:
        double probabilities[N_REGIMES]
        double expected_spot_vol
        int most_likely_regime
    
    void init_lut()
    void run_hmm_forward_pass_cpp(
        double r, double dt, SVCJParams* p_arr, double* t_mat, 
        double* last_p, Particle** clouds, HMMState* out) nogil

# The Python-facing HMM Engine
cdef class HMM_Engine:
    cdef SVCJParams* params_array
    cdef double* transition_matrix
    cdef double* current_probs
    cdef Particle** particle_clouds # Array of (Particle*)
    cdef double dt
    cdef int n_regimes
    
    def __init__(self, list regime_params, np.ndarray[double, ndim=2, mode='c'] trans_mat, double dt):
        init_lut() # Initialize the LUT
        
        self.n_regimes = len(regime_params)
        self.dt = dt
        
        # Allocate all C-memory
        self.params_array = <SVCJParams*> malloc(self.n_regimes * sizeof(SVCJParams))
        self.transition_matrix = <double*> malloc(self.n_regimes * self.n_regimes * sizeof(double))
        self.current_probs = <double*> malloc(self.n_regimes * sizeof(double))
        self.particle_clouds = <Particle**> malloc(self.n_regimes * sizeof(Particle*))
        
        for i, p in enumerate(regime_params):
            # Populate Physics
            self.params_array[i].mu = p.get('mu', 0.0)
            self.params_array[i].kappa = p['kappa']
            self.params_array[i].theta = p['theta']
            # ... and so on for all params ...
            self.params_array[i].sigma_v = p['sigma_v']
            self.params_array[i].rho = p['rho']
            self.params_array[i].lambda_j = p['lambda_j']
            self.params_array[i].mu_j = p.get('mu_j', -0.05)
            self.params_array[i].sigma_j = p.get('sigma_j', 0.1)

            # Initialize Particle Cloud for this regime
            self.particle_clouds[i] = <Particle*> malloc(N_PARTICLES * sizeof(Particle))
            for k in range(N_PARTICLES):
                self.particle_clouds[i][k].v = p['theta'] # Start at long run vol
                self.particle_clouds[i][k].w = 1.0 / N_PARTICLES
            
            # Initial Probabilities
            self.current_probs[i] = 1.0 / self.n_regimes
        
        # Populate Transition Matrix
        for r in range(self.n_regimes):
            for c in range(self.n_regimes):
                self.transition_matrix[r * self.n_regimes + c] = trans_mat[r, c]
            
    def __dealloc__(self):
        # Clean up all allocated memory
        free(self.params_array)
        free(self.transition_matrix)
        free(self.current_probs)
        for i in range(self.n_regimes):
            free(self.particle_clouds[i])
        free(self.particle_clouds)

    def update(self, double ret):
        cdef HMMState out_state
        
        run_hmm_forward_pass_cpp(
            ret, self.dt, self.params_array, self.transition_matrix,
            self.current_probs, self.particle_clouds, &out_state
        )
        
        # Persist new state
        for i in range(self.n_regimes):
            self.current_probs[i] = out_state.probabilities[i]
            
        # Return Python dict
        return {
            "probs": np.asarray(out_state.probabilities)[:self.n_regimes],
            "spot_vol_est": out_state.expected_spot_vol,
            "regime": out_state.most_likely_regime
        }