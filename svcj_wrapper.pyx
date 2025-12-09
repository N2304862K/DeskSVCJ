# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double theta, kappa, sigma_v, rho, lambda_j
    ctypedef struct FidelityMetrics:
        int win_impulse, win_gravity, is_valid
        double energy_ratio, residue_bias, ks_stat, hurst_exp
        double fit_theta, fit_kappa, fit_sigma_v, fit_rho, fit_lambda
    ctypedef struct InstantState:
        double current_spot_vol, innovation_z_score
    
    void run_fidelity_scan(double* ohlcv, int len, double dt, FidelityMetrics* out) nogil
    void run_instant_filter(double val, double vol, double avg, double dt, SVCJParams* p, double* st, InstantState* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def scan_fidelity_robust(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    if n < 200: return None # Need history for Hurst + Disjoint
    
    cdef FidelityMetrics m
    with nogil:
        run_fidelity_scan(&data[0,0], n, dt, &m)
        
    return {
        "windows": (m.win_impulse, m.win_gravity),
        "energy_ratio": m.energy_ratio,
        "residue_bias": m.residue_bias,
        "metrics": {
            "ks_stat": m.ks_stat,
            "hurst": m.hurst_exp
        },
        "is_valid": bool(m.is_valid),
        "params": {
            "theta": m.fit_theta, "kappa": m.fit_kappa, "sigma_v": m.fit_sigma_v,
            "rho": m.fit_rho, "lambda_j": m.fit_lambda
        }
    }

cdef class SpotMonitor:
    cdef SVCJParams params
    cdef double state_variance
    cdef double dt
    cdef double avg_volume
    
    def __init__(self, dict p, double dt, double avg_vol):
        self.params.theta = p['theta']
        self.params.kappa = p['kappa']
        self.params.sigma_v = p['sigma_v']
        self.params.rho = p['rho']
        self.params.lambda_j = p['lambda_j']
        self.state_variance = p['theta']
        self.dt = dt
        self.avg_volume = avg_vol
        
    def update(self, double price_now, double price_prev, double volume):
        cdef double ret = np.log(price_now / price_prev)
        cdef InstantState out
        run_instant_filter(ret, volume, self.avg_volume, self.dt, &self.params, &self.state_variance, &out)
        return {
            "z_score": out.innovation_z_score,
            "spot_vol": out.current_spot_vol
        }