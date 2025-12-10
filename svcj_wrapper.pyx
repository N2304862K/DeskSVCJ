# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    ctypedef struct EvolvingSystemState:
        pass
    ctypedef struct InstantMetrics:
        double expected_return, expected_vol, escape_velocity, surprise_index, swarm_entropy
        
    void initialize_system(double* ohlcv, int n, double dt, int n_p, EvolvingSystemState* out) nogil
    void run_system_step(EvolvingSystemState* state, double ret, double vol, InstantMetrics* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

# --- The Main Class: Evolving Particle Filter ---
cdef class EvolvingEngine:
    cdef EvolvingSystemState* state # Pointer to C-managed state
    
    def __cinit__(self, object ohlcv, double dt, int num_particles=1000):
        cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
        
        # Allocate C memory for the state struct
        self.state = <EvolvingSystemState*> malloc(sizeof(EvolvingSystemState))
        if self.state is NULL:
            raise MemoryError()
            
        # Call C initialization function
        initialize_system(&data[0,0], data.shape[0], dt, num_particles, self.state)
        
    def __dealloc__(self):
        if self.state is not NULL:
            free(self.state)
            
    def update(self, double new_ret, double new_vol):
        cdef InstantMetrics metrics
        
        # This function MUST be nogil for parallel execution if you had multiple
        # engines running. For one, it's fine.
        run_system_step(self.state, new_ret, new_vol, &metrics)
        
        return {
            "ev_ret": metrics.expected_return,
            "ev_vol": metrics.expected_vol,
            "escape": metrics.escape_velocity,
            "entropy": metrics.swarm_entropy
        }