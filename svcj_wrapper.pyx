# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct InstantState:
        double current_spot_vol, current_jump_prob, innovation_z_score
    ctypedef struct FidelityMetrics:
        int win_impulse, win_gravity, is_valid
        double energy_ratio, residue_bias, f_p_value, t_p_value
        double fit_theta, fit_kappa, fit_sigma_v, fit_rho, fit_lambda
    
    void run_fidelity_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out) nogil
    void run_instant_filter(double r, double dt, SVCJParams* p, double* state, InstantState* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def scan_fidelity(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    if n < 120: return None
    
    cdef FidelityMetrics m
    with nogil:
        run_fidelity_scan(&data[0,0], n, dt, &m)
        
    return {
        "windows": (m.win_impulse, m.win_gravity),
        "energy_ratio": m.energy_ratio,
        "residue_bias": m.residue_bias,
        "stats": {"f_p": m.f_p_value, "t_p": m.t_p_value},
        "is_valid": bool(m.is_valid),
        # THE COHERENCE PAYLOAD: Returns the exact fitted physics
        "params": {
            "theta": m.fit_theta,
            "kappa": m.fit_kappa,
            "sigma_v": m.fit_sigma_v,
            "rho": m.fit_rho,
            "lambda_j": m.fit_lambda
        }
    }

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