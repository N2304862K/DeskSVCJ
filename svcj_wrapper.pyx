# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct InstantState:
        double current_spot_vol, current_jump_prob, innovation_z_score, sprt_score
    ctypedef struct FidelityMetrics:
        int win_impulse, win_gravity, is_valid
        double energy_ratio, residue_bias, ks_stat, hurst_exp
        double fit_theta, fit_kappa, fit_sigma_v, fit_rho, fit_lambda
    
    void run_fidelity_scan(double* ohlcv, int len, double dt, FidelityMetrics* out) nogil
    void run_instant_filter(double val, double dt, double vscale, SVCJParams* p, double* st, double* sprt, InstantState* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def scan_fidelity_robust(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    if n < 150: return None
    cdef FidelityMetrics m
    with nogil:
        run_fidelity_scan(&data[0,0], n, dt, &m)
        
    return {
        "energy_ratio": m.energy_ratio,
        "residue_bias": m.residue_bias,
        "metrics": {"ks_stat": m.ks_stat, "hurst": m.hurst_exp},
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
    cdef double sprt_accum
    cdef int warmup
    
    def __init__(self, dict p, double dt, double avg_vol):
        self.params.mu=0; self.params.kappa=p['kappa']; self.params.theta=p['theta']
        self.params.sigma_v=p['sigma_v']; self.params.rho=p['rho']; self.params.lambda_j=p['lambda_j']
        self.params.mu_j=-0.05; self.params.sigma_j=0.05; 
        self.state_variance=p['theta']
        self.dt=dt; self.avg_volume=avg_vol
        self.sprt_accum=0
        self.warmup=30 # Cold start protection
        
    def update(self, double p_now, double p_prev, double volume):
        cdef double ret = np.log(p_now/p_prev)
        cdef double v_scale = (volume / self.avg_volume) if self.avg_volume > 0 else 1.0
        if v_scale < 0.1: v_scale = 0.1
        if v_scale > 10.0: v_scale = 10.0
        
        cdef InstantState out
        run_instant_filter(ret, self.dt, v_scale, &self.params, &self.state_variance, &self.sprt_accum, &out)
        
        # 7. Cold Start Logic
        if self.warmup > 0:
            self.warmup -= 1
            out.innovation_z_score = 0
            
        cdef np.ndarray[double, ndim=1] res = np.zeros(4)
        res[0] = out.innovation_z_score
        res[1] = out.current_spot_vol
        res[2] = out.sprt_score
        res[3] = out.current_jump_prob
        return res