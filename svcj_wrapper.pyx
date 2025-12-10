# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    ctypedef struct Particle:
        double kappa, theta, sigma_v, rho, lambda_j
        double v, weight
    
    ctypedef struct FilterStats:
        double spot_vol_mean, spot_vol_std, innovation_z
        double entropy, ess
        
    void generate_prior_swarm(double* ohlcv, int n, double dt, Particle* swarm) nogil
    void run_particle_filter_step(Particle* swarm, double ret, double dt, FilterStats* stats) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

# --- The Particle Filter Class ---
cdef class ParticleFilter:
    cdef Particle* swarm
    cdef public double dt  # <<< FIX: 'public' makes this accessible from Python
    cdef int swarm_size
    
    def __cinit__(self, object ohlcv, double dt):
        """
        Initializes and Seeds the Swarm.
        """
        cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
        cdef int n = data.shape[0]
        self.swarm_size = 5000
        
        self.swarm = <Particle*> malloc(self.swarm_size * sizeof(Particle))
        if self.swarm is NULL:
            raise MemoryError()
            
        self.dt = dt
        
        generate_prior_swarm(&data[0,0], n, dt, self.swarm)
        
    def __dealloc__(self):
        """
        Clean up C-memory when object is garbage collected.
        """
        if self.swarm is not NULL:
            free(self.swarm)
            
    def update(self, double price_now, double price_prev):
        """
        Runs one step of the filter.
        """
        cdef double ret = np.log(price_now / price_prev)
        cdef FilterStats stats
        
        # Run C-Core Step
        run_particle_filter_step(self.swarm, ret, self.dt, &stats)
        
        return {
            "spot_vol": stats.spot_vol_mean,
            "vol_uncertainty": stats.spot_vol_std,
            "z_score": stats.innovation_z,
            "ess": stats.ess,
            "entropy": stats.entropy
        }