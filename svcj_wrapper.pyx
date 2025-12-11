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
    ctypedef struct SwarmState:
        double ev_vol, mode_vol, ev_drift, entropy
        int collapsed
        
    void init_swarm(PhysicsParams* phys, Particle* swarm, double start_price) nogil
    void update_swarm(Particle* swarm, PhysicsParams* phys, 
                      double o, double h, double l, double c, 
                      double vol_ratio, double diurnal_factor, double momentum, double dt, 
                      SwarmState* out) nogil

cdef class IntradaySwarm:
    cdef PhysicsParams phys
    cdef Particle* swarm
    cdef double dt_base
    
    def __init__(self, dict params, double dt_base, double start_price):
        self._set_params(params)
        self.dt_base = dt_base
        self.swarm = <Particle*> malloc(2000 * sizeof(Particle))
        init_swarm(&self.phys, self.swarm, start_price)
        
    def __dealloc__(self):
        if self.swarm: free(self.swarm)
    
    cdef void _set_params(self, dict params):
        self.phys.kappa = params.get('kappa', 4.0)
        self.phys.theta = params.get('theta', 0.04)
        self.phys.sigma_v = params.get('sigma_v', 0.5)
        self.phys.lambda_j = params.get('lambda_j', 0.5)
        self.phys.mu_j = params.get('mu_j', -0.05)
        self.phys.sigma_j = params.get('sigma_j', 0.05)
            
    def update_tick(self, double o, double h, double l, double c, double v_ratio, double d_factor, double momentum):
        cdef SwarmState out
        with nogil:
            update_swarm(self.swarm, &self.phys, o, h, l, c, v_ratio, d_factor, momentum, self.dt_base, &out)
            
        return {
            "ev_vol": out.ev_vol,
            "mode_vol": out.mode_vol,
            "ev_drift": out.ev_drift,
            "entropy": out.entropy,
            "collapsed": bool(out.collapsed)
        }