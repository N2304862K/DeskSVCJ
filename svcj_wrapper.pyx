# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct InstantMetrics:
        double z_score, jerk, skew
        int recalibrate_flag
    
    int load_physics(const char* ticker, SVCJParams* p) nogil
    void run_full_calibration(const char* ticker, double* ohlcv, int n, double dt) nogil
    void initialize_tick_engine(SVCJParams* p, double dt) nogil
    void run_tick_update(double price, double volume, InstantMetrics* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

# --- The Self-Contained Engine Class ---
cdef class TickEngine:
    cdef str ticker
    cdef double dt
    cdef bint is_ready
    
    def __init__(self, str ticker, object historical_ohlcv, double dt):
        self.ticker = ticker
        self.dt = dt
        self.is_ready = False
        
        cdef SVCJParams p
        cdef bytes ticker_bytes = ticker.encode('utf-8')
        
        # 1. Attempt to load pre-calibrated model
        if load_physics(ticker_bytes, &p):
            initialize_tick_engine(&p, dt)
            self.is_ready = True
            print(f"[{ticker}] Physics loaded from disk. Monitor ready.")
        else:
            # 2. If no model exists, calibrate now
            print(f"[{ticker}] No physics file. Calibrating with provided history...")
            cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(historical_ohlcv)
            cdef int n = data.shape[0]
            
            run_full_calibration(ticker_bytes, &data[0,0], n, self.dt)
            self.is_ready = True
            print(f"[{ticker}] Calibration complete. Physics saved to {ticker}.bin.")
            
    def update(self, double price, double volume):
        """
        High-speed update. Returns Jerk & Skew.
        """
        if not self.is_ready: return None
        
        cdef InstantMetrics m
        run_tick_update(price, volume, &m)
        
        return {
            "z": m.z_score,
            "jerk": m.jerk,
            "skew": m.skew,
            "recalibrate_flag": bool(m.recalibrate_flag)
        }
        
    def force_recalibrate(self, object ohlcv):
        """
        Public method to trigger recalibration if model fails.
        """
        cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
        cdef int n = data.shape[0]
        cdef bytes ticker_bytes = self.ticker.encode('utf-8')
        
        run_full_calibration(ticker_bytes, &data[0,0], n, self.dt)
        print(f"[{self.ticker}] Recalibration forced. New physics saved.")