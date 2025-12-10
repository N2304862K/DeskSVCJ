# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct FidelityMetrics:
        int win_gravity, win_impulse, is_valid
        double theta_gravity, theta_impulse, energy_ratio
        double theta_std_err, param_z_score
        double ad_stat, hurst, residue_bias
    
    void run_fidelity_pipeline(double* ohlcv, int total_len, double dt, FidelityMetrics* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def scan_fidelity(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    
    if n < 200: return None # Need buffer for gravity scan
    
    cdef FidelityMetrics m
    with nogil:
        run_fidelity_pipeline(&data[0,0], n, dt, &m)
        
    return {
        "windows": (m.win_impulse, m.win_gravity),
        "physics": {
            "theta_grav": m.theta_gravity,
            "theta_imp": m.theta_impulse,
            "energy_ratio": m.energy_ratio
        },
        "stats": {
            "param_se": m.theta_std_err,
            "param_z": m.param_z_score,
            "ad_stat": m.ad_stat,
            "hurst": m.hurst
        },
        "residue_bias": m.residue_bias,
        "is_valid": bool(m.is_valid)
    }