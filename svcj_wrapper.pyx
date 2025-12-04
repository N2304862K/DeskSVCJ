# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct RegimeTestStats:
        double ll_constrained, ll_unconstrained, test_statistic, p_value
        int is_significant

    void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* sv, double* jp) nogil
    void perform_likelihood_ratio_test(double* ohlcv_long, int len_long, int len_short, double dt, RegimeTestStats* out) nogil
    double ukf_log_likelihood(double* ret, int n, double dt, SVCJParams* p, double* sv, double* jp, double proxy) nogil
    void compute_log_returns(double* ohlcv, int n, double* out) nogil

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
        "params": {
            "theta": p.theta, "kappa": p.kappa, "sigma_v": p.sigma_v,
            "rho": p.rho, "lambda_j": p.lambda_j
        },
        "spot_vol": sv,
        "jump_prob": jp
    }

def test_regime_break(object ohlcv, int long_window, int short_window, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    cdef RegimeTestStats s
    cdef int start = n - long_window
    
    perform_likelihood_ratio_test(&data[start,0], long_window, short_window, dt, &s)
    
    return {
        "stat": s.test_statistic,
        "p_value": s.p_value,
        "is_significant": True if s.is_significant else False
    }

def run_filter_instant(object ohlcv, double dt, dict params):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    cdef int n_ret = n - 1
    cdef np.ndarray[double, ndim=1] ret = np.zeros(n_ret)
    compute_log_returns(&data[0,0], n, &ret[0])
    
    cdef np.ndarray[double, ndim=1] sv = np.zeros(n_ret)
    cdef np.ndarray[double, ndim=1] jp = np.zeros(n_ret)
    cdef SVCJParams p
    p.mu=0; p.kappa=params['kappa']; p.theta=params['theta']
    p.sigma_v=params['sigma_v']; p.rho=params['rho']; p.lambda_j=params['lambda_j']
    p.mu_j=-0.05; p.sigma_j=0.05
    
    ukf_log_likelihood(&ret[0], n_ret, dt, &p, &sv[0], &jp[0], p.theta)
    return {"spot_vol": sv, "jump_prob": jp}