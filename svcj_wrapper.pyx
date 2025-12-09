# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct FidelityMetrics:
        int win_impulse, win_gravity, is_valid
        double energy_ratio, hurst_exponent
        double levene_p, mw_p, ks_p_vol
        double residue_median, fit_theta
    
    void run_enhanced_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def scan_enhanced_fidelity(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    
    # Need enough data for Impulse(30) + Gravity(min 60)
    if n < 100: return None
    
    cdef FidelityMetrics m
    with nogil:
        run_enhanced_scan(&data[0,0], n, dt, &m)
        
    return {
        "windows": (m.win_impulse, m.win_gravity),
        "metrics": {
            "energy_ratio": m.energy_ratio,
            "residue_median": m.residue_median,
            "hurst": m.hurst_exponent,
            "theta_struct": m.fit_theta
        },
        "stats": {
            "levene_p": m.levene_p,
            "mw_p": m.mw_p,
            "ks_vol_p": m.ks_p_vol
        },
        "is_valid": bool(m.is_valid)
    }