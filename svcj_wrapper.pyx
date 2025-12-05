# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j
    ctypedef struct FractalStats:
        double slope, intercept, r_squared, std_error, mean_theta
    
    void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* sv, double* jp) nogil
    void perform_fractal_test(double* ohlcv, int len, double dt, FractalStats* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def analyze_fractal_structure(object ohlcv, double dt):
    """
    Performs the Fractal Stability Test.
    Returns Linearity (R^2), Slope, and Stability Metrics.
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    
    # Needs at least enough data for the smallest window (30) * step (1.5)
    if n < 50: return None
    
    cdef FractalStats s
    perform_fractal_test(&data[0,0], n, dt, &s)
    
    return {
        "r_squared": s.r_squared,
        "slope": s.slope,
        "std_error": s.std_error,
        "mean_theta": s.mean_theta,
        "is_stable": (s.r_squared > 0.85)
    }

def fit_standalone(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    cdef int n_ret = n - 1
    cdef np.ndarray[double, ndim=1] sv = np.zeros(n_ret)
    cdef np.ndarray[double, ndim=1] jp = np.zeros(n_ret)
    cdef SVCJParams p
    
    optimize_svcj(&data[0,0], n, dt, &p, &sv[0], &jp[0])
    
    return {
        "params": {"theta": p.theta, "kappa": p.kappa, "sigma_v": p.sigma_v, "rho": p.rho, "lambda": p.lambda_j},
        "spot_vol": sv,
        "jump_prob": jp
    }