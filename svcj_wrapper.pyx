# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    ctypedef struct PhysicsParams:
        double kappa, theta, sigma_v, lambda_j, mu_j, sigma_j
    ctypedef struct Particle:
        double v, mu, rho, weight
    ctypedef struct TrendState:
        double trend_fast, trend_mid, trend_slow, coherence
    ctypedef struct SwarmState:
        double ev_vol, mode_vol, ev_drift, entropy, global_trend_coherence
        int collapsed
        
    void init_swarm(PhysicsParams* phys, Particle* swarm, double start_price, TrendState* ts) nogil
    void update_swarm(Particle* swarm, TrendState* ts, PhysicsParams* phys, 
                      double o, double h, double l, double c, 
                      double vol_ratio, double diurnal_factor, double dt, 
                      double prev_entropy,
                      SwarmState* out) nogil

cdef class IntradaySwarm:
    cdef PhysicsParams phys
    cdef Particle* swarm
    cdef TrendState* trend_state
    cdef double dt_base
    cdef double last_entropy # MEMORY
    
    def __init__(self, dict params, double dt_base, double start_price):
        self.phys.kappa = params.get('kappa', 4.0)
        self.phys.theta = params.get('theta', 0.04)
        self.phys.sigma_v = params.get('sigma_v', 0.5)
        self.phys.lambda_j = params.get('lambda_j', 0.5)
        self.phys.mu_j = params.get('mu_j', -0.05)
        self.phys.sigma_j = params.get('sigma_j', 0.05)
        
        self.dt_base = dt_base
        self.last_entropy = 0.5 # Start neutral
        self.swarm = <Particle*> malloc(2000 * sizeof(Particle))
        self.trend_state = <TrendState*> malloc(sizeof(TrendState))
        init_swarm(&self.phys, self.swarm, start_price, self.trend_state)
        
    def __dealloc__(self):
        if self.swarm: free(self.swarm)
        if self.trend_state: free(self.trend_state)
            
    def update_tick(self, double o, double h, double l, double c, double v_ratio, double d_factor):
        cdef SwarmState out
        
        with nogil:
            update_swarm(self.swarm, self.trend_state, &self.phys, 
                         o, h, l, c, v_ratio, d_factor, self.dt_base, 
                         self.last_entropy, &out)
        
        self.last_entropy = out.entropy # Update memory
        
        return {
            "ev_vol": out.ev_vol,
            "mode_vol": out.mode_vol,
            "ev_drift": out.ev_drift,
            "entropy": out.entropy,
            "collapsed": bool(out.collapsed),
            "trend_coherence": out.global_trend_coherence
        }