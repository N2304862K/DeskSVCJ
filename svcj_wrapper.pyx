# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    int N_PARTICLES
    
    ctypedef struct PhysicsParams:
        double kappa, theta, sigma_v, lambda_j, mu_j, sigma_j
    
    ctypedef struct Particle:
        double v, mu, rho, weight
        
    ctypedef struct SwarmState:
        double ev_vol, mode_vol, ev_drift, entropy
        
    void init_swarm(PhysicsParams* phys, Particle* swarm, double start_price) nogil
    void update_swarm(Particle* swarm, PhysicsParams* phys, 
                      double o, double h, double l, double c, 
                      double vol_ratio, double diurnal_factor, double dt, 
                      SwarmState* out) nogil

cdef class IntradaySwarm:
    cdef PhysicsParams phys
    cdef Particle* swarm
    cdef double dt_base
    
    def __init__(self, dict params, double dt_base, double start_price):
        self.phys.kappa = params.get('kappa', 4.0)
        self.phys.theta = params.get('theta', 0.04)
        self.phys.sigma_v = params.get('sigma_v', 0.3)
        self.phys.lambda_j = params.get('lambda_j', 0.5)
        self.phys.mu_j = params.get('mu_j', -0.05)
        self.phys.sigma_j = params.get('sigma_j', 0.05)
        
        self.dt_base = dt_base
        
        # Allocate Swarm Memory
        self.swarm = <Particle*> malloc(N_PARTICLES * sizeof(Particle))
        
        # Initialize
        init_swarm(&self.phys, self.swarm, start_price)
        
    def __dealloc__(self):
        if self.swarm:
            free(self.swarm)
            
    def update_tick(self, double o, double h, double l, double c, double vol_ratio, double diurnal_factor):
        """
        Feeds the swarm a new bar.
        vol_ratio: CurrentVol / AvgVol (Volume Clock)
        diurnal_factor: Expected Intraday Vol Multiplier (Seasonality)
        """
        cdef SwarmState out
        
        with nogil:
            update_swarm(self.swarm, &self.phys, 
                         o, h, l, c, 
                         vol_ratio, diurnal_factor, self.dt_base, 
                         &out)
            
        return {
            "ev_vol": out.ev_vol,       # Mean (Risk)
            "mode_vol": out.mode_vol,   # Mode (Robust State)
            "ev_drift": out.ev_drift,   # Trend Strength
            "entropy": out.entropy      # Confidence (Low = Action)
        }