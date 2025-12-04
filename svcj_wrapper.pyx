# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
import pandas as pd # Used for time inference

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct SVCJGreeks:
        double delta, gamma, vega, theta_decay

    void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* sv, double* jp) nogil
    void calc_svcj_greeks(double s0, double K, double T, double r, SVCJParams* p, double v, int type, SVCJGreeks* out) nogil
    double ukf_log_likelihood(double* ret, int n, double dt, SVCJParams* p, double* sv, double* jp, double proxy) nogil

# --- 1. Dynamic Time Inference ---
def infer_dt_annualized(object index_obj):
    """
    Calculates annualized DT from a Pandas DatetimeIndex.
    Handles Daily, Hourly, Minute data automatically.
    """
    if len(index_obj) < 2: return 1.0/252.0
    
    # Median diff in seconds
    series = pd.Series(index_obj)
    diffs = series.diff().dropna().dt.total_seconds()
    median_sec = diffs.median()
    
    # Financial Year in Seconds (assuming 252 days * 6.5 hours trading)
    # This aligns intraday volatility to annualized volatility
    # Daily: 86400 raw? No, markets close. 
    # Standard practice: Effective trading time per year ~ 252 * 24h (if crypto) or 252 * 6.5h (if stocks)
    # SIMPLIFICATION: We map the observed frequency to fractions of 252 days.
    
    # If gap is > 20 hours (Daily data)
    if median_sec > 70000: 
        return 1.0/252.0
    
    # If gap is ~1 hour
    if 3000 < median_sec < 4000:
        return 1.0/(252.0 * 6.5) # Assuming 6.5h trading day
        
    # If gap is ~5 min (300 sec)
    if 250 < median_sec < 350:
        return 1.0/(252.0 * 78.0) # 78 5-min bars per day
        
    # Fallback: Ratio of seconds
    seconds_per_trading_year = 252.0 * 6.5 * 60 * 60
    return median_sec / seconds_per_trading_year

# --- 2. Standalone Structure Engine (No Options) ---
def analyze_structure_standalone(object df_ohlcv, list windows):
    """
    Detects Regime Breaks using only OHLCV.
    Returns: Tension Metrics, Parameters, Spot States.
    """
    # 1. Infer DT
    cdef double dt = infer_dt_annualized(df_ohlcv.index)
    
    # 2. Prepare Data
    cdef np.ndarray[double, ndim=2, mode='c'] data = np.ascontiguousarray(df_ohlcv.values, dtype=np.float64)
    cdef int total_len = data.shape[0]
    cdef int n_wins = len(windows)
    
    cdef np.ndarray[double, ndim=2] params_out = np.zeros((n_wins, 8))
    cdef SVCJParams p
    cdef int i, w_len, start
    
    # 3. Horizon Surface Analysis
    for i in range(n_wins):
        w_len = windows[i]
        if w_len > total_len: continue
        start = total_len - w_len
        
        # Optimize on specific window history
        optimize_svcj(&data[start, 0], w_len, dt, &p, NULL, NULL)
        
        params_out[i, 0] = p.theta
        params_out[i, 1] = p.kappa
        params_out[i, 2] = p.sigma_v
        params_out[i, 3] = p.lambda_j
        params_out[i, 4] = p.rho
        params_out[i, 5] = p.mu
        
    return {
        "dt": dt,
        "horizon_params": params_out # [Window, Param]
    }

# --- 3. Intraday Filter ---
def run_filter(object prices, dict params, double dt):
    """
    Runs UKF on price array with fixed params and specific dt.
    """
    cdef np.ndarray[double, ndim=1, mode='c'] c_p = np.ascontiguousarray(prices, dtype=np.float64)
    cdef int n = c_p.shape[0]
    cdef int n_ret = n - 1
    
    cdef np.ndarray[double, ndim=1] ret = np.diff(np.log(c_p))
    cdef np.ndarray[double, ndim=1] sv = np.zeros(n_ret)
    cdef np.ndarray[double, ndim=1] jp = np.zeros(n_ret)
    
    cdef SVCJParams p
    p.mu=params['mu']; p.kappa=params['kappa']; p.theta=params['theta']
    p.sigma_v=params['sigma_v']; p.rho=params['rho']; p.lambda_j=params['lambda_j']
    p.mu_j=params.get('mu_j',0); p.sigma_j=params.get('sigma_j',0.05)
    
    ukf_log_likelihood(&ret[0], n_ret, dt, &p, &sv[0], &jp[0], p.theta)
    
    return {"spot_vol": sv, "jump_prob": jp}