# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct InstantState:
        double current_spot_vol, current_jump_prob, innovation_z_score
    ctypedef struct FidelityMetrics:
        int win_impulse, win_gravity, is_valid
        double energy_ratio, residue_bias, ks_stat, hurst_exp
        double fit_theta, fit_kappa, fit_sigma_v, fit_rho, fit_lambda
    ctypedef struct VoVPoint:
        int window
        double sigma_v, theta
    
    void run_fidelity_scan(double* ohlcv, int len, double dt, FidelityMetrics* out) nogil
    void run_vov_scan(double* ohlcv, int len, double dt, int step, VoVPoint* buf, int max) nogil
    void run_instant_filter(double v, double vol, double avg, double dt, SVCJParams* p, double* st, double* sprt, InstantState* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

# --- Scanner ---
def scan_fidelity_spectrum(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    if n < 200: return None
    
    # 1. VoV Scan (Find Natural Freq)
    cdef int step = 5
    cdef int max_steps = int((n-60)/step)+2
    cdef VoVPoint* buf = <VoVPoint*> malloc(max_steps*sizeof(VoVPoint))
    
    with nogil:
        run_vov_scan(&data[0,0], n, dt, step, buf, max_steps)
        
    windows=[]; sigmas=[];
    for i in range(max_steps):
        if buf[i].window==0: break
        windows.append(buf[i].window)
        sigmas.append(buf[i].sigma_v)
    free(buf)
    
    if not sigmas: return None
    min_idx = np.argmin(sigmas)
    nat_win = windows[min_idx]
    
    # 2. Fidelity
    cdef FidelityMetrics m
    with nogil:
        # Note: We actually pass full data, internal C does disjoint split
        run_fidelity_scan(&data[0,0], n, dt, &m)
        
    return {
        "natural_window": nat_win,
        "energy_ratio": m.energy_ratio,
        "residue_bias": m.residue_bias,
        "is_valid": bool(m.is_valid),
        "params": {
            "theta": m.fit_theta, "kappa": m.fit_kappa, "sigma_v": m.fit_sigma_v,
            "rho": m.fit_rho, "lambda_j": m.fit_lambda
        }
    }

# --- Improvement 7 & 10: SpotMonitor ---
cdef class SpotMonitor:
    cdef SVCJParams params
    cdef double state_variance
    cdef double dt
    cdef double avg_volume
    cdef double sprt_accumulator
    cdef int warmup_counter
    
    def __init__(self, dict p, double dt, double avg_vol=1000000, int warmup=30):
        self.params.mu=0; self.params.kappa=p['kappa']; self.params.theta=p['theta']
        self.params.sigma_v=p['sigma_v']; self.params.rho=p['rho']; self.params.lambda_j=p['lambda_j']
        self.params.mu_j=-0.05; self.params.sigma_j=0.05
        self.state_variance=p['theta']
        self.dt=dt
        self.avg_volume=avg_vol
        self.sprt_accumulator=0.0
        self.warmup_counter=warmup
        
    def update(self, double price_now, double price_prev, double volume):
        # No-Alloc Return Buffer
        cdef np.ndarray[double, ndim=1] ret_buf = np.zeros(4)
        cdef InstantState out
        
        cdef double r = np.log(price_now / price_prev)
        run_instant_filter(r, volume, self.avg_volume, self.dt, &self.params, &self.state_variance, &self.sprt_accumulator, &out)
        
        if self.warmup_counter > 0:
            self.warmup_counter -= 1
            out.innovation_z_score = 0.0 # Silence during warmup
            
        ret_buf[0] = out.innovation_z_score
        ret_buf[1] = out.current_spot_vol
        ret_buf[2] = out.current_jump_prob
        ret_buf[3] = self.sprt_accumulator
        
        return ret_buf