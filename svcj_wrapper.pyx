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
    double ukf_log_likelihood(double* ret, int n, double dt, SVCJParams* p, double* sv, double* jp, double proxy) nogil
    void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

# --- Feature 1: Unified Unified Fit (Time Agnostic) ---
def fit_and_filter(object ohlcv, double dt):
    """
    Fits parameters and runs filter.
    dt: Annualized time step (e.g. 1/252 for daily)
    """
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
            "rho": p.rho, "lambda_j": p.lambda_j
        },
        "spot_vol": spot_vol,
        "jump_prob": jump_prob
    }

# --- Feature 2: Dynamic Surface Tension (Gradient Detection) ---
def analyze_multiscale_gradient(object ohlcv, double dt):
    """
    Auto-detects surface breaking by fitting on Log-Scale windows.
    Returns [WindowSize, Theta] matrix.
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int total_len = data.shape[0]
    
    # Generate Log-Scale Windows down to 30
    windows = []
    curr = total_len
    while curr >= 30:
        windows.append(curr)
        curr = int(curr / 1.5)
    windows = windows[::-1] 
    
    cdef int n_wins = len(windows)
    cdef np.ndarray[double, ndim=2] res = np.zeros((n_wins, 2))
    cdef SVCJParams p
    cdef int i, w, start_idx
    
    for i in range(n_wins):
        w = windows[i]
        start_idx = total_len - w
        # Fit on this window slice
        optimize_svcj(&data[start_idx, 0], w, dt, &p, NULL, NULL)
        res[i, 0] = w
        res[i, 1] = p.theta
        
    return res

# --- Feature 3: Greeks ---
def get_greeks(double s0, double K, double T, double r, dict params, double spot_vol, int type):
    cdef SVCJParams p
    p.lambda_j=params['lambda_j']; p.mu_j=params.get('mu_j', -0.05); 
    p.sigma_j=params.get('sigma_j', 0.05); p.mu=r;
    
    cdef SVCJGreeks g
    calc_svcj_greeks(s0, K, T, r, &p, spot_vol, type, &g)
    return {"delta": g.delta, "gamma": g.gamma, "vega": g.vega}

# --- Feature 4: Instantaneous Filter (No Re-Calib) ---
def run_filter_fixed(object ohlcv, double dt, dict params):
    cdef np.ndarray[double, ndim=2, mode='c'] c_ohlcv = _sanitize(ohlcv)
    cdef int n = c_ohlcv.shape[0]
    cdef int n_ret = n - 1
    
    cdef np.ndarray[double, ndim=1] ret = np.zeros(n_ret)
    compute_log_returns(&c_ohlcv[0,0], n, &ret[0])
    
    cdef np.ndarray[double, ndim=1] sv = np.zeros(n_ret)
    cdef np.ndarray[double, ndim=1] jp = np.zeros(n_ret)
    
    cdef SVCJParams p
    p.mu=0; p.kappa=params['kappa']; p.theta=params['theta']
    p.sigma_v=params['sigma_v']; p.rho=params['rho']; p.lambda_j=params['lambda_j']
    p.mu_j=params.get('mu_j', -0.05); p.sigma_j=params.get('sigma_j', 0.05)
    
    ukf_log_likelihood(&ret[0], n_ret, dt, &p, &sv[0], &jp[0], p.theta)
    return {"spot_vol": sv, "jump_prob": jp}