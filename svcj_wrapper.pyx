# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct SVCJGreeks:
        double delta, gamma, vega, theta_decay
    ctypedef struct TermStructStats:
        double slope, curvature, short_term_theta, long_term_theta, divergence

    void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* sv, double* jp) nogil
    void calculate_structural_gradient(double* ohlcv, int len, double dt, TermStructStats* out) nogil
    void calc_svcj_greeks(double s0, double K, double T, double r, SVCJParams* p, double v, int type, SVCJGreeks* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def get_structural_gradient(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] c_ohlcv = _sanitize(ohlcv)
    cdef TermStructStats s
    calculate_structural_gradient(&c_ohlcv[0,0], c_ohlcv.shape[0], dt, &s)
    return {
        "slope": s.slope, "short_theta": s.short_term_theta,
        "long_theta": s.long_term_theta, "divergence": s.divergence
    }

def fit_standalone(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] c_ohlcv = _sanitize(ohlcv)
    cdef int n = c_ohlcv.shape[0]
    cdef int n_ret = n - 1
    cdef np.ndarray[double, ndim=1] sv = np.zeros(n_ret)
    cdef np.ndarray[double, ndim=1] jp = np.zeros(n_ret)
    cdef SVCJParams p
    
    optimize_svcj(&c_ohlcv[0,0], n, dt, &p, &sv[0], &jp[0])
    
    return {
        "params": {
            "theta": p.theta, "kappa": p.kappa, "sigma_v": p.sigma_v,
            "rho": p.rho, "lambda_j": p.lambda_j
        },
        "spot_vol": sv, "jump_prob": jp
    }

def get_greeks(double s0, double K, double T, double r, dict params, double spot_vol, int type):
    cdef SVCJParams p
    p.lambda_j=params['lambda_j']; p.mu_j=params.get('mu_j', -0.05); 
    p.sigma_j=params.get('sigma_j', 0.05); p.mu=r;
    cdef SVCJGreeks g
    calc_svcj_greeks(s0, K, T, r, &p, spot_vol, type, &g)
    return {"delta": g.delta, "gamma": g.gamma, "vega": g.vega}