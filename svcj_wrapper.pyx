# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    
    void clean_returns(double* returns, int n) nogil
    void optimize_svcj(double* ohlcv, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob) nogil
    void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double spot_vol, double* out_prices) nogil

# Helper: Ensure OHLCV is (N, 5) C-Contiguous
cdef np.ndarray[double, ndim=2, mode='c'] _sanitize_ohlcv(object input_data):
    cdef np.ndarray[double, ndim=2] arr = np.asarray(input_data, dtype=np.float64)
    if arr.shape[1] != 5:
        raise ValueError("Input must be OHLCV matrix with 5 columns")
    return np.ascontiguousarray(arr)

# Helper: Process Option Chain to Columns
cdef tuple _process_chain(object option_chain):
    cdef np.ndarray arr = np.asarray(option_chain, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("Option chain must be (N, 4) matrix: [Strike, Expiry, Type, Price]")
    
    cdef np.ndarray[double, ndim=1, mode='c'] ks = np.ascontiguousarray(arr[:, 0])
    cdef np.ndarray[double, ndim=1, mode='c'] ts = np.ascontiguousarray(arr[:, 1])
    cdef np.ndarray[int, ndim=1, mode='c'] types = np.ascontiguousarray(arr[:, 2].astype(np.int32))
    return ks, ts, types, arr.shape[0]

# --- Method 1: Asset-Specific Option Adjusted ---
def generate_asset_option_adjusted(object ohlcv, double s0, object option_chain):
    """
    ohlcv: (N, 5) Matrix [Open, High, Low, Close, Volume]
    option_chain: (M, 4) Matrix [Strike, T, Type, Price]
    """
    cdef np.ndarray[double, ndim=2, mode='c'] c_ohlcv = _sanitize_ohlcv(ohlcv)
    cdef int n = c_ohlcv.shape[0]
    
    # Process Options
    ks, ts, types, n_opts = _process_chain(option_chain)
    cdef np.ndarray[double, ndim=1, mode='c'] c_ks = ks
    cdef np.ndarray[double, ndim=1, mode='c'] c_ts = ts
    cdef np.ndarray[int, ndim=1, mode='c'] c_types = types
    
    # Output Arrays (Size n-1 for vol because returns logic, but we map to n inside or align)
    # The C logic computes returns internally (size n-1). 
    # We allocate (n-1) for results to match returns index.
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n-1)
    cdef np.ndarray[double, ndim=1] jump_prob = np.zeros(n-1)
    cdef np.ndarray[double, ndim=1] model_prices = np.zeros(n_opts)
    
    cdef SVCJParams p
    
    # Run Optimization using OHLCV
    optimize_svcj(&c_ohlcv[0, 0], n, &p, &spot_vol[0], &jump_prob[0])
    
    # Price using *Current* Spot Vol (Last Element)
    price_option_chain(s0, &c_ks[0], &c_ts[0], &c_types[0], n_opts, &p, spot_vol[n-2], &model_prices[0])
    
    return {
        "params": {
            "kappa": p.kappa, "theta": p.theta, "sigma_v": p.sigma_v, 
            "rho": p.rho, "lambda_j": p.lambda_j, "mu_j": p.mu_j, "sigma_j": p.sigma_j
        },
        "spot_vol": spot_vol,
        "jump_prob": jump_prob,
        "model_prices": model_prices
    }

# --- Method 2: Market Wide Rolling ---
def analyze_market_rolling(object market_ohlcv_dict, int window):
    # This is tricky for "Matrix In" if assets have different inputs.
    # We assume input is a 3D Tensor: (Assets, Time, 5)
    cdef np.ndarray[double, ndim=3] data = np.asarray(market_ohlcv_dict, dtype=np.float64)
    # Check layout: (Assets, Time, 5) vs (Time, Assets, 5)?
    # Assuming (Assets, Time, 5) for C-contiguity per asset
    
    if data.shape[2] != 5:
        raise ValueError("3D Input must be (Assets, Time, 5)")

    cdef int n_assets = data.shape[0]
    cdef int n_days = data.shape[1]
    cdef int n_windows = n_days - window
    
    cdef np.ndarray[double, ndim=3] results = np.zeros((n_assets, n_windows, 5))
    cdef SVCJParams p
    cdef int i, w
    
    # Need to be careful passing 3D slice to 2D pointer
    # &data[i, w, 0] gives start of OHLCV for Asset i at Time w
    
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

# --- Method 3: Market Wide Current ---
def analyze_market_current(object market_ohlcv_tensor):
    # Input: (Assets, Time, 5)
    cdef np.ndarray[double, ndim=3, mode='c'] data = np.ascontiguousarray(market_ohlcv_tensor, dtype=np.float64)
    cdef int n_assets = data.shape[0]
    cdef int n_days = data.shape[1]
    cdef int n_ret = n_days - 1
    
    cdef np.ndarray[double, ndim=2] out_spot = np.zeros((n_assets, n_ret))
    cdef np.ndarray[double, ndim=2] out_jump = np.zeros((n_assets, n_ret))
    cdef np.ndarray[double, ndim=2] out_params = np.zeros((n_assets, 8))
    
    cdef int i
    cdef SVCJParams p
    
    with nogil:
        for i in prange(n_assets):
            optimize_svcj(&data[i, 0, 0], n_days, &p, &out_spot[i, 0], &out_jump[i, 0])
            out_params[i, 0] = p.kappa
            out_params[i, 1] = p.theta
            out_params[i, 2] = p.sigma_v
            out_params[i, 3] = p.rho
            out_params[i, 4] = p.lambda_j
            out_params[i, 5] = p.mu_j
            out_params[i, 6] = p.sigma_j
            out_params[i, 7] = p.mu
            
    return {"spot_vol": out_spot.T, "jump_prob": out_jump.T, "params": out_params}

# --- Method 4: Residue Analysis ---
def generate_residue_analysis(object ohlcv):
    cdef np.ndarray[double, ndim=2, mode='c'] c_ohlcv = _sanitize_ohlcv(ohlcv)
    cdef int n = c_ohlcv.shape[0]
    cdef int n_ret = n - 1
    
    cdef np.ndarray[double, ndim=1] residues = np.zeros(n_ret)
    cdef SVCJParams p
    
    # We need returns to calc residue
    cdef np.ndarray[double, ndim=1] returns = np.zeros(n_ret)
    
    # Optimize
    optimize_svcj(&c_ohlcv[0, 0], n, &p, NULL, NULL)
    
    # Re-compute returns in Python for residue calc (or expose C func)
    # Simple log diff of Close (Col 3)
    cdef int t
    for t in range(n_ret):
        returns[t] = np.log(c_ohlcv[t+1, 3] / c_ohlcv[t, 3])
        residues[t] = returns[t] - (p.mu * (1.0/252.0))
        
    return residues