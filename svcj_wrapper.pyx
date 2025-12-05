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
    ctypedef struct SVCJGreeks:
        double delta, gamma, vega, theta_decay
    
    void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* sv, double* jp) nogil
    void perform_likelihood_test(double* ohlcv, int len_long, int len_short, double dt, RegimeStats* out) nogil
    void calc_greeks(double s0, double K, double T, double r, SVCJParams* p, double v, int type, SVCJGreeks* out) nogil

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
    
    # Validation: Ensure we have enough data for the Long Window
    # If not, truncate the request to available data
    if n < long_window:
        long_window = n 
    
    # Must have enough for Short Window + some buffer
    if long_window <= short_window + 5:
        raise ValueError("Data too short for regime test")
        
    cdef int start = n - long_window
    cdef RegimeStats s
    
    perform_likelihood_test(&data[start,0], long_window, short_window, dt, &s)
    
    return {
        "stat": s.statistic,
        "p_value": s.p_value,
        "significant": bool(s.significant),
        "ll_ratio": s.ll_null / s.ll_alt if s.ll_alt != 0 else 0
    }

def get_greeks(double s0, double K, double T, double r, dict params, double spot_vol, int type):
    cdef SVCJParams p
    p.lambda_j=params['lambda']; p.mu_j=-0.05; p.sigma_j=0.05; p.mu=r;
    cdef SVCJGreeks g
    calc_greeks(s0, K, T, r, &p, spot_vol, type, &g)
    return {"delta": g.delta, "gamma": g.gamma, "vega": g.vega}