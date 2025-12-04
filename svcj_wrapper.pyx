# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct SnapshotStats:
        double ll_ratio, p_value, short_theta, long_theta, divergence

    void optimize_snapshot_raw(double* ohlcv, int n, double dt, SVCJParams* p) nogil
    void run_snapshot_test(double* ohlcv, int w_long, int w_short, double dt, SnapshotStats* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def analyze_instant_snapshot(object ohlcv, int w_long, int w_short, double dt):
    """
    Runs the Multi-Start Snapshot Engine.
    No Rolling. Fits NOW based on Long vs Short windows.
    Returns statistical validity of the break.
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    if n < w_long: raise ValueError("Data too short")
    
    cdef SnapshotStats stats
    cdef int start_idx = n - w_long
    
    # Run C-Core Pipeline
    run_snapshot_test(&data[start_idx, 0], w_long, w_short, dt, &stats)
    
    return {
        "p_value": stats.p_value,
        "likelihood_ratio": stats.ll_ratio,
        "divergence": stats.divergence,
        "long_theta": stats.long_theta,
        "short_theta": stats.short_theta
    }

def get_current_params(object ohlcv, double dt):
    """
    Returns the raw unconstrained parameters for the current window.
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    cdef SVCJParams p
    
    optimize_snapshot_raw(&data[0,0], n, dt, &p)
    
    return {
        "theta": p.theta, "lambda": p.lambda_j, "kappa": p.kappa, 
        "sigma_v": p.sigma_v, "rho": p.rho
    }