# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct SpectralPoint:
        int window_len
        double theta, spot_vol, residue_sum
    
    void run_spectral_scan(double* ohlcv, int total_len, int* windows, int n_windows, double dt, SpectralPoint* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def analyze_physics_spectrum(object ohlcv, double dt):
    """
    1. Generates log-scale windows based on data length.
    2. Scans physics for all windows.
    3. Identifies 'Gravity Wall' (Stable Theta) and 'Impulse' (Kinetic).
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    
    # 1. Generate Log-Scale Windows (Python List -> C Array)
    # Start small (30), go up to N, step 1.4
    win_list = []
    curr = 30.0
    while curr <= n:
        win_list.append(int(curr))
        curr *= 1.4
    
    if not win_list: return None # Data too short
    
    cdef int n_wins = len(win_list)
    cdef np.ndarray[int, ndim=1, mode='c'] c_wins = np.array(win_list, dtype=np.int32)
    cdef np.ndarray[double, ndim=2] res_matrix = np.zeros((n_wins, 4)) # [Len, Theta, Spot, Residue]
    
    # Create C-Struct Array wrapper
    # We allocate memory in Python to pass to C? No, C output is array of structs.
    # Actually, simpler to just map struct array to numpy in Cython after call.
    cdef SpectralPoint* results = <SpectralPoint*> malloc(n_wins * sizeof(SpectralPoint))
    
    try:
        run_spectral_scan(&data[0,0], n, &c_wins[0], n_wins, dt, results)
        
        # 2. Extract Results to Python
        points = []
        for i in range(n_wins):
            points.append({
                "window": results[i].window_len,
                "theta": results[i].theta,
                "spot": results[i].spot_vol,
                "residue": results[i].residue_sum
            })
    finally:
        free(results)
        
    # 3. Find Gravity Wall (Data-Derived Logic)
    # Gravity = The window where Theta changes least relative to neighbors.
    # (Plateau in the Theta vs Window curve)
    
    # Calculate gradient of theta
    thetas = np.array([p['theta'] for p in points])
    grads = np.abs(np.gradient(thetas))
    
    # Find index of min gradient (Most stable structure)
    gravity_idx = np.argmin(grads)
    gravity_point = points[gravity_idx]
    
    # 4. Find Impulse (Shortest valid window)
    impulse_point = points[0]
    
    # 5. Calculate Physics
    energy_ratio = (impulse_point['spot']**2) / gravity_point['theta']
    bias = impulse_point['residue']
    
    return {
        "gravity_window": gravity_point['window'],
        "gravity_theta": gravity_point['theta'],
        "impulse_window": impulse_point['window'],
        "energy_ratio": energy_ratio,
        "residue_bias": bias,
        "is_breakout": (energy_ratio > 1.2 and bias > 0)
    }

from libc.stdlib cimport malloc, free