# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct FidelityMetrics:
        int win_impulse, win_gravity, is_valid
        double energy_ratio, residue_bias, f_stat, f_p_value, t_stat, t_p_value
    
    void run_fidelity_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def analyze_fidelity(object ohlcv, double dt):
    """
    Runs the Frequency Separation + Statistical Fidelity Check.
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    
    # Must have enough data for 4x30 = 120 bars minimum
    if n < 120: return None
    
    cdef FidelityMetrics m
    
    # Release GIL for Heavy Scan
    with nogil:
        run_fidelity_scan(&data[0,0], n, dt, &m)
        
    if m.win_gravity == 0: return None # Error in C
    
    return {
        "windows": (m.win_impulse, m.win_gravity),
        "energy_ratio": m.energy_ratio,
        "residue_bias": m.residue_bias,
        "stats": {
            "f_stat": m.f_stat,
            "f_p_val": m.f_p_value,
            "t_stat": m.t_stat,
            "t_p_val": m.t_p_value
        },
        "is_high_fidelity": bool(m.is_valid)
    }