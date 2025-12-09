# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct FidelityMetrics:
        double energy_ratio, residue_bias, hurst_exponent
        double ks_stat, realized_f_stat, f_p_value, jb_p
        int is_valid
    
    void run_fidelity_scan_advanced(double* ohlcv, int total_len, int w_grav, int w_imp, double dt, FidelityMetrics* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def scan_advanced_fidelity(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    
    if n < 160: return None
    
    cdef FidelityMetrics m
    with nogil:
        # Grav=120, Imp=30
        run_fidelity_scan_advanced(&data[0,0], n, 120, 30, dt, &m)
        
    return {
        "physics": {
            "energy_ratio": m.energy_ratio,
            "bias": m.residue_bias,
            "hurst": m.hurst_exponent
        },
        "stats": {
            "f_stat": m.realized_f_stat,
            "f_p": m.f_p_value,
            "ks_stat": m.ks_stat,
            "jb_p": m.jb_p
        },
        "is_valid": bool(m.is_valid)
    }