# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j
    ctypedef struct InstantMetrics:
        double z_score, jerk, skew
        int recalibrate_flag
    
    int load_physics(const char* ticker, SVCJParams* p) nogil
    void run_full_calibration(const char* ticker, double* ohlcv, int n, double dt) nogil
    void initialize_tick_engine(SVCJParams* p, double dt) nogil
    void run_tick_update(double price, double volume, InstantMetrics* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

# The Main Interface
cdef class InstantMonitor:
    cdef str ticker
    cdef double dt
    cdef bint is_ready
    
    def __init__(self, str ticker, double dt):
        self.ticker = ticker
        self.dt = dt
        self.is_ready = False
        
        cdef SVCJParams p
        cdef bytes ticker_bytes = ticker.encode('utf-8')
        
        # Try to load physics from file
        if load_physics(ticker_bytes, &p):
            initialize_tick_engine(&p, dt)
            self.is_ready = True
            print(f"[{ticker}] Physics loaded from disk. Monitor ready.")
        else:
            print(f"[{ticker}] No physics file found. CALIBRATION REQUIRED.")
            
    def calibrate(self, object ohlcv):
        """
        Runs the heavy calibration and saves the model to disk.
        """
        cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
        cdef int n = data.shape[0]
        cdef bytes ticker_bytes = self.ticker.encode('utf-8')
        
        run_full_calibration(ticker_bytes, &data[0,0], n, self.dt)
        self.is_ready = True
        print(f"[{self.ticker}] Calibration complete. Physics saved.")
        
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
            "recalibrate": bool(m.recalibrate_flag)
        }