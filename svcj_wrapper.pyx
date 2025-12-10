# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j
    ctypedef struct CausalStats:
        double max_deviation, p_value, hurst_exponent, residue_bias, energy_ratio
        int is_breakout
    
    void fit_gravity_physics(double* ohlcv, int n, double dt, SVCJParams* out) nogil
    void test_causal_cone(double* imp, int n, double dt, SVCJParams* grav, CausalStats* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def analyze_causal_structure(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    
    # Define Windows (Disjoint)
    cdef int w_imp = 30
    cdef int w_grav = 150
    if n < w_grav + w_imp: return None
    
    cdef int start_grav = n - w_imp - w_grav
    
    # 1. Fit Gravity
    cdef SVCJParams physics
    fit_gravity_physics(&data[start_grav, 0], w_grav, dt, &physics)
    
    # 2. Test Impulse
    cdef np.ndarray[double, ndim=1] imp_prices = data[n-w_imp:, 3].copy()
    cdef CausalStats stats
    
    test_causal_cone(&imp_prices[0], w_imp, dt, &physics, &stats)
    
    return {
        "physics": { "theta": physics.theta, "drift_mu": physics.mu },
        "breakout": {
            "max_z": stats.max_deviation,
            "p_value": stats.p_value,
            "is_valid": bool(stats.is_breakout),
            "hurst": stats.hurst_exponent,
            "bias": stats.residue_bias,
            "energy": stats.energy_ratio
        }
    }