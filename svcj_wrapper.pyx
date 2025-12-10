# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct FrequencyResult:
        int natural_window
        double min_sigma_v
    ctypedef struct CausalStats:
        double max_deviation, p_value, drift_term, vol_term
        int is_breakout, break_index
    
    void run_vov_spectrum_scan(double* ohlcv, int len, double dt, int step, FrequencyResult* out) nogil
    void fit_gravity_physics(double* ohlcv, int n, double dt, SVCJParams* out) nogil
    void test_causal_cone(double* imp, int n, double dt, SVCJParams* grav, CausalStats* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def analyze_causal_structure(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    if n < 100: return None
    
    # 1. Scan for Natural Frequency (Gravity Window)
    cdef FrequencyResult freq
    with nogil:
        run_vov_spectrum_scan(&data[0,0], n, dt, 10, &freq)
    
    cdef int w_grav = freq.natural_window
    if w_grav < 60: w_grav = 60 # Safety floor
    
    # 2. Fit Physics on Gravity Window (Last N bars)
    # We define Gravity as the structure LEADING UP TO the Impulse.
    # Impulse = Last 30 bars. Gravity = Bars before that.
    cdef int w_imp = 30
    cdef int start_grav = n - w_imp - w_grav
    if start_grav < 0: start_grav = 0
    
    cdef SVCJParams physics
    fit_gravity_physics(&data[start_grav, 0], w_grav, dt, &physics)
    
    # 3. Test Causal Cone on Impulse (Last 30 bars)
    # We pass the CLOSE prices (Column 3) of the impulse window
    cdef np.ndarray[double, ndim=1] imp_prices = data[n-w_imp:, 3].copy()
    cdef CausalStats stats
    
    test_causal_cone(&imp_prices[0], w_imp, dt, &physics, &stats)
    
    return {
        "gravity_window": w_grav,
        "physics": {
            "theta": physics.theta,
            "drift_mu": physics.mu,
            "sigma_v": physics.sigma_v
        },
        "breakout": {
            "max_z": stats.max_deviation,
            "p_value": stats.p_value,
            "is_valid": bool(stats.is_breakout),
            "break_bar": stats.break_index
        }
    }