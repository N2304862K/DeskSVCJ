# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import prange

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct InstantState:
        double current_spot_vol, current_jump_prob, innovation_z_score
    ctypedef struct FidelityMetrics:
        double energy_ratio, residue_bias, f_p_value, t_p_value
        int is_valid
        double fit_theta, fit_kappa, fit_sigma_v, fit_rho, fit_lambda
    
    void run_fidelity_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out) nogil
    void run_instant_filter(double r, double dt, SVCJParams* p, double* state, InstantState* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

# --- MARKET SCALE ENGINE ---
def scan_market_batch(object tensor_3d, double dt):
    """
    Input: (Assets, Time, 5) Tensor
    Output: Dictionary of Vectorized Results (aligned by Asset Index)
    """
    cdef np.ndarray[double, ndim=3, mode='c'] data = np.ascontiguousarray(tensor_3d, dtype=np.float64)
    cdef int n_assets = data.shape[0]
    cdef int n_bars = data.shape[1]
    
    # Pre-allocate C-Struct Array for results
    cdef FidelityMetrics* results = <FidelityMetrics*> malloc(n_assets * sizeof(FidelityMetrics))
    
    # 1. Parallel Execution (GIL Released)
    # The 'Physics' of every asset is solved independently on separate cores
    cdef int i
    with nogil:
        for i in prange(n_assets, schedule='dynamic'):
            # Pass pointer to the start of this asset's OHLCV block
            # data[i, 0, 0] is the address
            run_fidelity_scan(&data[i, 0, 0], n_bars, dt, &results[i])
            
    # 2. Unpack to Python (Vectorized)
    # We construct arrays to return clean columns for a DataFrame
    cdef np.ndarray[double, ndim=1] energy = np.zeros(n_assets)
    cdef np.ndarray[double, ndim=1] bias = np.zeros(n_assets)
    cdef np.ndarray[double, ndim=1] fp = np.zeros(n_assets)
    cdef np.ndarray[double, ndim=1] tp = np.zeros(n_assets)
    cdef np.ndarray[int, ndim=1] valid = np.zeros(n_assets, dtype=np.int32)
    
    # Physics Payload Arrays
    cdef np.ndarray[double, ndim=1] theta = np.zeros(n_assets)
    cdef np.ndarray[double, ndim=1] kappa = np.zeros(n_assets)
    cdef np.ndarray[double, ndim=1] sigma_v = np.zeros(n_assets)
    cdef np.ndarray[double, ndim=1] rho = np.zeros(n_assets)
    cdef np.ndarray[double, ndim=1] lam = np.zeros(n_assets)
    
    for i in range(n_assets):
        energy[i] = results[i].energy_ratio
        bias[i] = results[i].residue_bias
        fp[i] = results[i].f_p_value
        tp[i] = results[i].t_p_value
        valid[i] = results[i].is_valid
        
        theta[i] = results[i].fit_theta
        kappa[i] = results[i].fit_kappa
        sigma_v[i] = results[i].fit_sigma_v
        rho[i] = results[i].fit_rho
        lam[i] = results[i].fit_lambda
        
    free(results)
    
    return {
        "energy_ratio": energy,
        "residue_bias": bias,
        "f_p_value": fp,
        "t_p_value": tp,
        "is_valid": valid,
        "params": {
            "theta": theta, "kappa": kappa, "sigma_v": sigma_v, "rho": rho, "lambda": lam
        }
    }

# --- Instant Monitor (For Selected Assets) ---
cdef class SpotMonitor:
    cdef SVCJParams params
    cdef double state_variance
    cdef double dt
    
    def __init__(self, dict p, double dt):
        self.params.mu = 0
        self.params.kappa = p['kappa']
        self.params.theta = p['theta']
        self.params.sigma_v = p['sigma_v']
        self.params.rho = p['rho']
        self.params.lambda_j = p['lambda_j']
        self.params.mu_j = -0.05
        self.params.sigma_j = 0.05
        self.state_variance = p['theta']
        self.dt = dt
        
    def update(self, double price_now, double price_prev):
        cdef double ret = np.log(price_now / price_prev)
        cdef InstantState out
        run_instant_filter(ret, self.dt, &self.params, &self.state_variance, &out)
        return {
            "z_score": out.innovation_z_score,
            "spot_vol": out.current_spot_vol,
            "jump_prob": out.current_jump_prob
        }