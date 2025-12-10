# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport strcpy

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    ctypedef struct TickState:
        pass
    ctypedef struct InstantMetrics:
        double z_score, jerk, skew
        int needs_recalibration
    
    int save_physics(const char* ticker, SVCJParams* p) nogil
    int load_physics(const char* ticker, SVCJParams* p) nogil
    void optimize_svcj_volume(double* ohlcv, int n, double dt, SVCJParams* p) nogil
    void init_tick_state(TickState* state, SVCJParams* p) nogil
    void run_tick_update(double price, double vol, double dt, SVCJParams* p, TickState* state, InstantMetrics* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

# --- The Self-Organizing Engine ---
cdef class TickEngine:
    cdef SVCJParams params
    cdef TickState state
    cdef double dt
    cdef char ticker[256]
    
    def __init__(self, str ticker_str, object initial_ohlcv, double dt):
        self.dt = dt
        
        # Convert Python string to C char*
        cdef bytes ticker_bytes = ticker_str.encode('UTF-8')
        strcpy(self.ticker, ticker_bytes)
        
        # Try to load physics, if fail, calibrate.
        if not load_physics(self.ticker, &self.params):
            print(f"[{ticker_str}] No saved physics found. Calibrating...")
            self.recalibrate(initial_ohlcv)
        else:
            print(f"[{ticker_str}] Loaded physics from cache.")
            
        init_tick_state(&self.state, &self.params)

    def recalibrate(self, object ohlcv):
        cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
        cdef int n = data.shape[0]
        
        # Heavy compute, release GIL
        with nogil:
            optimize_svcj_volume(&data[0,0], n, self.dt, &self.params)
            save_physics(self.ticker, &self.params)
            
        print(f"   > Recalibrated & saved. Theta={self.params.theta:.4f}")
        init_tick_state(&self.state, &self.params) # Reset buffers

    def update(self, double price, double vol, object ohlcv_hist=None):
        cdef InstantMetrics m
        
        run_tick_update(price, vol, self.dt, &self.params, &self.state, &m)
        
        # Internal failure detection
        if m.needs_recalibration == 1:
            print(f"   >>> MODEL COHERENCE FAILURE (KS Test). Triggering recalibration.")
            if ohlcv_hist is not None:
                self.recalibrate(ohlcv_hist)
            # After recalibration, the *next* tick will be valid.
            # Return a 'safe' state for this tick.
            return {"z_score": 0, "jerk": 0, "skew": 0, "recalibrated": True}

        return {
            "z_score": m.z_score,
            "jerk": m.jerk,
            "skew": m.skew,
            "recalibrated": False
        }