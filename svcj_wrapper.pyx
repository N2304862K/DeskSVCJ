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
    ctypedef struct VoVPoint:
        int window
        double sigma_v, theta
    ctypedef struct FidelityMetrics:
        int win_impulse, win_gravity, is_valid
        double energy_ratio, residue_bias, f_p_value, t_p_value, lb_p_value
        double fit_theta, fit_kappa, fit_sigma_v, fit_rho, fit_lambda
    ctypedef struct SVCJGreeks:
        double delta, gamma, vega
    
    void run_vov_scan(double* ohlcv, int total_len, double dt, int step, VoVPoint* out_buffer, int max_steps) nogil
    void run_fidelity_check(double* ohlcv, int total_len, int win_grav, int win_imp, double dt, FidelityMetrics* out) nogil
    void run_instant_filter(double r, double dt, SVCJParams* p, double* state, InstantState* out) nogil
    void calc_greeks(double s0, double K, double T, double r, SVCJParams* p, double v, int type, SVCJGreeks* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

# --- VoV Spectrum Orchestrator ---
def scan_fidelity_spectrum(object ohlcv, double dt):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    if n < 100: return None
    
    # 1. Run VoV Scan
    cdef int step = 5
    cdef int max_steps = int((n - 30) / step) + 2
    cdef VoVPoint* buf = <VoVPoint*> malloc(max_steps * sizeof(VoVPoint))
    
    with nogil:
        run_vov_scan(&data[0,0], n, dt, step, buf, max_steps)
        
    # Find Natural Frequency (Min Sigma_V) in Python for flexibility
    windows = []
    sigmas = []
    cdef int i
    for i in range(max_steps):
        if buf[i].window == 0: break
        windows.append(buf[i].window)
        sigmas.append(buf[i].sigma_v)
    free(buf)
    
    if not sigmas: return None
    
    # Simple Min Find
    min_idx = np.argmin(sigmas)
    natural_window = windows[min_idx]
    
    # 2. Run Fidelity Check
    cdef FidelityMetrics m
    # Fixed Impulse window 30
    run_fidelity_check(&data[0,0], n, natural_window, 30, dt, &m)
    
    return {
        "natural_window": natural_window,
        "energy_ratio": m.energy_ratio,
        "residue_bias": m.residue_bias,
        "is_valid": bool(m.is_valid),
        "stats": {
            "f_p": m.f_p_value, "t_p": m.t_p_value, "lb_p": m.lb_p_value
        },
        # EXPORT PHYSICS
        "params": {
            "theta": m.fit_theta,
            "kappa": m.fit_kappa,
            "sigma_v": m.fit_sigma_v,
            "rho": m.fit_rho,
            "lambda_j": m.fit_lambda
        }
    }

# --- Instant Monitor ---
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

# --- Greeks ---
def get_greeks(double s0, double K, double T, double r, dict params, double spot_vol, int type):
    cdef SVCJParams p
    p.lambda_j=params['lambda_j']; p.mu_j=-0.05; p.sigma_j=0.05; p.mu=r;
    cdef SVCJGreeks g
    calc_greeks(s0, K, T, r, &p, spot_vol, type, &g)
    return {"delta": g.delta, "gamma": g.gamma, "vega": g.vega}