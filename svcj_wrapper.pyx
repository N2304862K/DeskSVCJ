# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct SVCJGreeks:
        double delta, gamma, vega, theta_decay
    ctypedef struct RegimeTestStats:
        double ll_constrained, ll_unconstrained, test_statistic, p_value
        int is_significant

    void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* sv, double* jp) nogil
    void perform_likelihood_ratio_test(double* ohlcv_long, int len_long, int len_short, double dt, RegimeTestStats* out) nogil
    void calc_svcj_greeks(double s0, double K, double T, double r, SVCJParams* p, double v, int type, SVCJGreeks* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

# --- Feature 1: Statistical Regime Test (Likelihood Ratio) ---
def test_regime_break(object ohlcv, int long_window, int short_window, double dt):
    """
    Performs Wilks' Likelihood Ratio Test.
    Returns: P-Value (Prob that Short data is NOT a break).
    Small P-Value (< 0.05) = Statistically Significant Break.
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int total_len = data.shape[0]
    
    # We use the LAST long_window points
    if total_len < long_window:
        raise ValueError("Not enough data for Long Window")
        
    cdef RegimeTestStats stats
    
    # Pass pointer to the start of the Long Window (end - long_window)
    cdef int start_idx = total_len - long_window
    
    perform_likelihood_ratio_test(&data[start_idx, 0], long_window, short_window, dt, &stats)
    
    return {
        "stat": stats.test_statistic,
        "p_value": stats.p_value,
        "is_significant": True if stats.is_significant == 1 else False,
        "ll_ratio": stats.ll_constrained / stats.ll_unconstrained # Fit Quality Ratio
    }

# --- Feature 2: Standalone Fit ---
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

# --- Feature 3: Greeks ---
def get_greeks(double s0, double K, double T, double r, dict params, double spot_vol, int type):
    cdef SVCJParams p
    p.lambda_j=params['lambda_j']; p.mu_j=-0.05; p.sigma_j=0.05; p.mu=r;
    cdef SVCJGreeks g
    calc_svcj_greeks(s0, K, T, r, &p, spot_vol, type, &g)
    return {"delta": g.delta, "gamma": g.gamma, "vega": g.vega}