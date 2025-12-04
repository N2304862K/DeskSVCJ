# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct RegimeMetrics:
        double ll_ratio_stat, p_value, divergence, short_theta, long_theta

    void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p) nogil
    double ukf_log_likelihood(double* ret, int n, double dt, SVCJParams* p, double* sv, double* jp, double proxy) nogil
    void compute_log_returns(double* ohlcv, int n, double* out) nogil
    void run_structural_test(double* ohlcv, int l_long, int l_short, double dt, RegimeMetrics* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

# --- Pipeline Function 1: Structural Test ---
def assess_structural_integrity(object ohlcv, int w_long, int w_short, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    if n < w_long: raise ValueError("Data shorter than Long Window")
    
    cdef RegimeMetrics res
    # Pointer to end of data array minus long window
    cdef int start_idx = n - w_long
    
    run_structural_test(&data[start_idx, 0], w_long, w_short, dt, &res)
    
    return {
        "p_value": res.p_value,
        "divergence": res.divergence,
        "short_theta": res.short_theta,
        "long_theta": res.long_theta
    }

# --- Pipeline Function 2: Instantaneous State ---
def filter_instantaneous_state(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    cdef int n_ret = n - 1
    
    # 1. Fit Params
    cdef SVCJParams p
    optimize_svcj(&data[0,0], n, dt, &p)
    
    # 2. Run Filter
    cdef np.ndarray[double, ndim=1] ret = np.zeros(n_ret)
    compute_log_returns(&data[0,0], n, &ret[0])
    
    cdef np.ndarray[double, ndim=1] sv = np.zeros(n_ret)
    cdef np.ndarray[double, ndim=1] jp = np.zeros(n_ret)
    
    ukf_log_likelihood(&ret[0], n_ret, dt, &p, &sv[0], &jp[0], p.theta)
    
    return {
        "params": {
            "theta": p.theta, "lambda": p.lambda_j, "kappa": p.kappa, "rho": p.rho
        },
        "spot_vol": sv,
        "jump_prob": jp,
        # Return last residue for Z-Score calc
        "last_ret": ret[n_ret-1],
        "last_drift": (p.mu - 0.5*sv[n_ret-1]*sv[n_ret-1])*dt
    }