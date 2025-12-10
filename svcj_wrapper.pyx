# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    ctypedef struct BreakoutSignal:
        double ks_stat, p_value, energy_ratio, drift_z_score
        int is_breakout
        
    void run_hierarchical_scan(double* ohlcv, int len_long, int len_short, double dt, int n_particles, BreakoutSignal* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def scan_hierarchical(object ohlcv, int win_long, int win_short, double dt, int n_particles=500):
    """
    Runs the Hierarchical Swarm Analysis (Gravity vs Impulse).
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    
    if n < win_long: win_long = n
    if win_long < win_short + 10: return None
        
    cdef BreakoutSignal s
    
    # Pass pointer to START of long window
    cdef int start_idx = n - win_long
    
    with nogil:
        run_hierarchical_scan(&data[start_idx, 0], win_long, win_short, dt, n_particles, &s)
        
    return {
        "ks_stat": s.ks_stat,
        "p_value": s.p_value,
        "energy_ratio": s.energy_ratio,
        "drift": s.drift_z_score,
        "is_breakout": bool(s.is_breakout)
    }