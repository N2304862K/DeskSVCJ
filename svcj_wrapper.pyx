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
    ctypedef struct GradientStats:
        double slope, curvature, short_theta, long_theta

    void compute_log_returns(double* ohlcv, int n, double* out) nogil
    void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* sv, double* jp) nogil
    void perform_likelihood_ratio_test(double* ohlcv, int l_long, int l_short, double dt, RegimeTestStats* out) nogil
    void calculate_structural_gradient(double* ohlcv, int len, double dt, GradientStats* out) nogil
    double ukf_log_likelihood(double* r, int n, double dt, SVCJParams* p, double* sv, double* jp, double proxy) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

# --- Feature 1: Model-Model Statistical Test ---
def test_regime_break(object ohlcv, int l_long, int l_short, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    if n < l_long: raise ValueError("Insufficient Data")
    
    cdef RegimeTestStats s
    # Pass pointer to end of data minus window
    perform_likelihood_ratio_test(&data[n-l_long, 0], l_long, l_short, dt, &s)
    
    return {
        "stat": s.test_statistic,
        "p_value": s.p_value,
        "significant": True if s.is_significant else False
    }

# --- Feature 2: Gradient Physics ---
def get_energy_gradient(object ohlcv, int window, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    cdef GradientStats g
    
    # Analyze the LAST 'window' points
    if n > window:
        calculate_structural_gradient(&data[n-window, 0], window, dt, &g)
    else:
        calculate_structural_gradient(&data[0,0], n, dt, &g)
        
    return {
        "slope": g.slope,
        "short_theta": g.short_theta,
        "long_theta": g.long_theta,
        "divergence": (g.short_theta - g.long_theta)/g.long_theta
    }

# --- Feature 3: Instantaneous Filter (Fixed Params) ---
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
    p.mu_j=params.get('mu_j',0); p.sigma_j=params.get('sigma_j',0.05)
    
    ukf_log_likelihood(&ret[0], n_ret, dt, &p, &sv[0], &jp[0], p.theta)
    return {"spot_vol": sv, "jump_prob": jp}

# --- Feature 4: Calibration ---
def fit_model(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef SVCJParams p
    optimize_svcj(&data[0,0], data.shape[0], dt, &p, NULL, NULL)
    return {
        "theta": p.theta, "kappa": p.kappa, "sigma_v": p.sigma_v,
        "rho": p.rho, "lambda_j": p.lambda_j, "mu_j": p.mu_j, "sigma_j": p.sigma_j
    }