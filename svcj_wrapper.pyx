# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct FidelityMetrics:
        double energy_ratio, residue_bias, hurst_exponent
        double ks_stat, levene_p, jb_p
        int is_valid
    
    void run_fidelity_scan_native(double* ohlcv, int total_len, int w_grav, int w_imp, double dt, FidelityMetrics* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def scan_fidelity(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    
    # Needs sufficient history: Grav(150) + Imp(30)
    if n < 200: return None
    
    cdef FidelityMetrics m
    
    with nogil:
        # Fixed Windows: Gravity 150 (Stable), Impulse 30 (Fast)
        run_fidelity_scan_native(&data[0,0], n, 150, 30, dt, &m)
        
    return {
        "energy_ratio": m.energy_ratio,
        "bias": m.residue_bias,
        "hurst": m.hurst_exponent,
        "stats": {
            "ks_stat": m.ks_stat,
            "levene_p": m.levene_p,
            "jb_p": m.jb_p
        },
        "is_valid": bool(m.is_valid)
    }