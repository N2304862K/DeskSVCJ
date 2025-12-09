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
        # EXACT ORDER FROM C HEADER
        double energy_ratio
        double residue_median
        double levene_p, mw_p, ks_ret_p, ks_vol_p
        
        double fit_theta, fit_kappa, fit_sigma_v, fit_rho, fit_lambda
        
        int win_impulse, win_gravity, is_valid
    
    void run_full_audit_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out) nogil
    void run_instant_filter_vol(double ret, double vol, double avg, double dt, SVCJParams* p, double* state, InstantState* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def scan_audit_fidelity(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    if n < 150: return None
    
    cdef FidelityMetrics m
    with nogil:
        run_full_audit_scan(&data[0,0], n, dt, &m)
        
    return {
        "energy_ratio": m.energy_ratio,
        "residue_median": m.residue_median,
        "is_valid": bool(m.is_valid),
        "stats": {
            "levene_p": m.levene_p, "mw_p": m.mw_p, 
            "ks_ret_p": m.ks_ret_p, "ks_vol_p": m.ks_vol_p
        },
        "params": {
            "theta": m.fit_theta, "kappa": m.fit_kappa, "sigma_v": m.fit_sigma_v,
            "rho": m.fit_rho, "lambda_j": m.fit_lambda
        }
    }

cdef class VolumeSpotMonitor:
    cdef SVCJParams params
    cdef double state_variance
    cdef double dt
    cdef double avg_vol
    
    def __init__(self, dict p, double dt, double avg_vol):
        self.params.mu=0; self.params.kappa=p['kappa']; self.params.theta=p['theta']
        self.params.sigma_v=p['sigma_v']; self.params.rho=p['rho']; self.params.lambda_j=p['lambda_j']
        self.params.mu_j=-0.05; self.params.sigma_j=0.05; self.state_variance=p['theta']; self.dt=dt
        self.avg_vol = avg_vol
        
    def update(self, double p_now, double p_prev, double vol_now):
        cdef double ret = np.log(p_now/p_prev)
        cdef InstantState out
        run_instant_filter_vol(ret, vol_now, self.avg_vol, self.dt, &self.params, &self.state_variance, &out)
        return {"z_score": out.innovation_z_score, "spot_vol": out.current_spot_vol, "jump_prob": out.current_jump_prob}