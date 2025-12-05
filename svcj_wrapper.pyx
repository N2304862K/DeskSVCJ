# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    ctypedef struct VoVPoint:
        int window
        double sigma_v, theta
    ctypedef struct FidelityMetrics:
        int win_impulse, win_gravity, is_valid
        double energy_ratio, residue_bias, f_p_value, t_p_value, lb_p_value, lb_stat
    
    void run_vov_scan(double* ohlcv, int total_len, double dt, int step, VoVPoint* out_buffer, int max_steps) nogil
    void run_fidelity_check(double* ohlcv, int total_len, int win_grav, int win_imp, double dt, FidelityMetrics* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def analyze_vov_structure(object ohlcv, double dt):
    """
    1. Scans VoV Spectrum.
    2. Finds Natural Frequency (Min Sigma_V).
    3. Runs Fidelity Check against Natural Frequency.
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    
    if n < 100: return None
    
    # 1. Run Spectrum Scan
    # Step size 5 to be dense enough
    cdef int step = 5
    cdef int max_steps = int((n - 60) / step) + 2
    if max_steps <= 0: return None
    
    cdef VoVPoint* spec_buf = <VoVPoint*> malloc(max_steps * sizeof(VoVPoint))
    
    with nogil:
        run_vov_scan(&data[0,0], n, dt, step, spec_buf, max_steps)
        
    # 2. Python Side: Find Natural Frequency
    # Extract
    windows = []
    sigmas = []
    thetas = []
    
    cdef int i
    for i in range(max_steps):
        if spec_buf[i].window == 0: break
        windows.append(spec_buf[i].window)
        sigmas.append(spec_buf[i].sigma_v)
        thetas.append(spec_buf[i].theta)
        
    free(spec_buf)
    
    if not sigmas: return None
    
    # Find Local Minimum of Sigma_V
    # We smooth slightly to avoid noise
    sigs = np.array(sigmas)
    # Natural Freq is the window with lowest Vol of Vol (Most stable regime)
    min_idx = np.argmin(sigs)
    natural_window = windows[min_idx]
    natural_sigma = sigs[min_idx]
    
    # 3. Run Fidelity Check
    # Impulse is fixed at 30 bars (Stat minimum)
    cdef FidelityMetrics fm
    run_fidelity_check(&data[0,0], n, natural_window, 30, dt, &fm)
    
    return {
        "natural_window": natural_window,
        "stability_sigma": natural_sigma,
        "energy_ratio": fm.energy_ratio,
        "residue_bias": fm.residue_bias,
        "stats": {
            "f_p": fm.f_p_value,
            "t_p": fm.t_p_value,
            "lb_p": fm.lb_p_value,
            "lb_stat": fm.lb_stat
        },
        "is_valid_breakout": bool(fm.is_valid)
    }