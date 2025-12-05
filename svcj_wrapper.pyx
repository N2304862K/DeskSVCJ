# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j
    ctypedef struct RegimeStats:
        double ll_null, ll_alt, statistic, p_value
        int significant
    
    void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* sv, double* jp) nogil
    void perform_likelihood_test(double* ohlcv, int len_long, int len_short, double dt, RegimeStats* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

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

def test_regime_break(object ohlcv, int long_window, int short_window, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    # Use last N points
    cdef int start = n - long_window
    if start < 0: raise ValueError("Data too short")
    
    cdef RegimeStats s
    perform_likelihood_test(&data[start,0], long_window, short_window, dt, &s)
    
    return {
        "stat": s.statistic,
        "p_value": s.p_value,
        "significant": bool(s.significant),
        "ll_ratio": s.ll_null / s.ll_alt
    }