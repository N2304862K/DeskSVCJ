# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from cython.parallel import prange

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct SVCJGreeks:
        double delta, gamma, vega, theta_decay
    
    void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* sv, double* jp) nogil
    void calc_svcj_greeks(double s0, double K, double T, double r, SVCJParams* p, double v, int type, SVCJGreeks* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def fit_standalone(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] c_ohlcv = _sanitize(ohlcv)
    cdef int n = c_ohlcv.shape[0]
    cdef int n_ret = n - 1
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n_ret)
    cdef np.ndarray[double, ndim=1] jump_prob = np.zeros(n_ret)
    cdef SVCJParams p
    optimize_svcj(&c_ohlcv[0, 0], n, dt, &p, &spot_vol[0], &jump_prob[0])
    return {
        "params": {
            "theta": p.theta, "kappa": p.kappa, "sigma_v": p.sigma_v,
            "rho": p.rho, "lambda_j": p.lambda_j, "mu_j": p.mu_j, "sigma_j": p.sigma_j
        },
        "spot_vol": spot_vol, "jump_prob": jump_prob
    }

# --- Gradient Generation (Log Scale Windows) ---
def analyze_log_scale_windows(object ohlcv, double dt):
    """ Returns Matrix [WindowSize, Theta] for Gradient Regression """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int total_len = data.shape[0]
    windows = []
    curr = total_len
    # Generate geometric series of windows
    while curr >= 30:
        windows.append(curr)
        curr = int(curr / 1.4) # 1.4 step size for density
    windows = windows[::-1] 
    
    cdef int n_wins = len(windows)
    cdef np.ndarray[double, ndim=2] res = np.zeros((n_wins, 2))
    cdef SVCJParams p
    cdef int i, w, start_idx
    
    for i in range(n_wins):
        w = windows[i]
        start_idx = total_len - w
        optimize_svcj(&data[start_idx, 0], w, dt, &p, NULL, NULL)
        res[i, 0] = w
        res[i, 1] = p.theta
    return res

def get_full_greeks(double s0, double K, double T, double r, dict params, double spot_vol, int type):
    cdef SVCJParams p
    p.lambda_j=params['lambda_j']; p.mu_j=params.get('mu_j', -0.05); 
    p.sigma_j=params.get('sigma_j', 0.05); p.mu=r;
    cdef SVCJGreeks g
    calc_svcj_greeks(s0, K, T, r, &p, spot_vol, type, &g)
    return {"delta": g.delta, "gamma": g.gamma, "vega": g.vega}