# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct BreakoutStats:
        double energy_ratio, drift_impulse, jump_dominance, residue_bias
        int is_breakout
    
    void calculate_breakout_physics(double* ohlcv, int len_long, int len_short, double dt, BreakoutStats* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def analyze_breakout_force(object ohlcv, int win_long, int win_short, double dt):
    """
    Calculates Breakout Physics (Energy, Drift, Direction).
    Target: Identify Upward Structural Breaks.
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    
    # Handle lengths
    if n < win_long: win_long = n
    if win_long < win_short + 10: 
        # Fallback if data very short
        return None 
        
    cdef int start_idx = n - win_long
    cdef BreakoutStats s
    
    calculate_breakout_physics(&data[start_idx, 0], win_long, win_short, dt, &s)
    
    return {
        "energy_ratio": s.energy_ratio,
        "drift_impulse": s.drift_impulse,
        "jump_dominance": s.jump_dominance,
        "residue_bias": s.residue_bias,
        "is_breakout": bool(s.is_breakout)
    }