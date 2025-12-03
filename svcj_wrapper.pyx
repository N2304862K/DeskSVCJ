# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    
    void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) nogil
    void optimize_svcj(double* ohlcv, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob) nogil
    double ukf_log_likelihood(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob, double realized_theta_proxy) nogil
    void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double spot_vol, double* out_prices) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize_ohlcv(object input_data):
    cdef np.ndarray[double, ndim=2] arr = np.asarray(input_data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError("Input must be OHLCV matrix (Time, 5)")
    return np.ascontiguousarray(arr)

cdef tuple _process_chain(object option_chain):
    cdef np.ndarray arr = np.asarray(option_chain, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("Option chain must be (N, 4) matrix")
    cdef np.ndarray[double, ndim=1, mode='c'] ks = np.ascontiguousarray(arr[:, 0].ravel())
    cdef np.ndarray[double, ndim=1, mode='c'] ts = np.ascontiguousarray(arr[:, 1].ravel())
    cdef np.ndarray[int, ndim=1, mode='c'] types = np.ascontiguousarray(arr[:, 2].ravel().astype(np.int32))
    return ks, ts, types, arr.shape[0]

# --- New: Instantaneous Filter (Fixed Params) ---
def run_filter_fixed(object ohlcv, dict params):
    cdef np.ndarray[double, ndim=2, mode='c'] c_ohlcv = _sanitize_ohlcv(ohlcv)
    cdef int n = c_ohlcv.shape[0]
    cdef int n_ret = n - 1
    
    cdef np.ndarray[double, ndim=1] returns = np.zeros(n_ret)
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n_ret)
    cdef np.ndarray[double, ndim=1] jump_prob = np.zeros(n_ret)
    
    # Pre-calc returns
    compute_log_returns(&c_ohlcv[0,0], n, &returns[0])
    
    cdef SVCJParams p
    p.mu = params.get('mu', 0.0)
    p.kappa = params['kappa']
    p.theta = params['theta']
    p.sigma_v = params['sigma_v']
    p.rho = params['rho']
    p.lambda_j = params['lambda_j']
    p.mu_j = params.get('mu_j', 0.0)
    p.sigma_j = params.get('sigma_j', 0.05)
    
    # Run Filter directly (No Optimization)
    ukf_log_likelihood(&returns[0], n_ret, &p, &spot_vol[0], &jump_prob[0], p.theta)
    
    return {"spot_vol": spot_vol, "jump_prob": jump_prob}

# --- Asset Specific ---
def generate_asset_option_adjusted(object ohlcv, double s0, object option_chain):
    cdef np.ndarray[double, ndim=2, mode='c'] c_ohlcv = _sanitize_ohlcv(ohlcv)
    cdef int n = c_ohlcv.shape[0]
    ks, ts, types, n_opts = _process_chain(option_chain)
    cdef np.ndarray[double, ndim=1, mode='c'] c_ks = ks
    cdef np.ndarray[double, ndim=1, mode='c'] c_ts = ts
    cdef np.ndarray[int, ndim=1, mode='c'] c_types = types
    
    cdef int n_ret = n - 1
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n_ret)
    cdef np.ndarray[double, ndim=1] jump_prob = np.zeros(n_ret)
    cdef np.ndarray[double, ndim=1] model_prices = np.zeros(n_opts)
    
    cdef SVCJParams p
    optimize_svcj(&c_ohlcv[0, 0], n, &p, &spot_vol[0], &jump_prob[0])
    price_option_chain(s0, &c_ks[0], &c_ts[0], &c_types[0], n_opts, &p, spot_vol[n_ret-1], &model_prices[0])
    
    return {
        "params": {
            "kappa": p.kappa, "theta": p.theta, "sigma_v": p.sigma_v, 
            "rho": p.rho, "lambda_j": p.lambda_j
        },
        "spot_vol": spot_vol,
        "jump_prob": jump_prob,
        "model_prices": model_prices
    }

# --- Market Rolling ---
def analyze_market_rolling(object market_ohlcv_tensor, int window):
    cdef np.ndarray[double, ndim=3, mode='c'] data = np.ascontiguousarray(market_ohlcv_tensor, dtype=np.float64)
    if data.shape[2] != 5: raise ValueError("Input must be (Assets, Time, 5)")
    cdef int n_assets = data.shape[0]
    cdef int n_days = data.shape[1]
    cdef int n_windows = n_days - window
    if n_windows < 1: return None
    cdef np.ndarray[double, ndim=3] results = np.zeros((n_assets, n_windows, 5))
    cdef SVCJParams p
    cdef int i, w
    with nogil:
        for i in prange(n_assets, schedule='dynamic'):
            for w in range(n_windows):
                optimize_svcj(&data[i, w, 0], window, &p, NULL, NULL)
                results[i, w, 0] = p.theta
                results[i, w, 1] = p.kappa
                results[i, w, 2] = p.sigma_v
                results[i, w, 3] = p.rho
                results[i, w, 4] = p.lambda_j
    return results