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
    ctypedef struct IMMState:
        double prob_bull, prob_bear, prob_neutral
        double agg_vol, agg_drift, entropy
        
    void init_imm(PhysicsParams* phys, Particle* p, double start) nogil
    void update_imm(Particle* p, PhysicsParams* phys, 
                    double o, double h, double l, double c, 
                    double vr, double df, double dt, 
                    IMMState* out) nogil

cdef class IMMFilter:
    cdef PhysicsParams phys
    cdef Particle* particles
    cdef double dt_base
    
    def __init__(self, dict params, double dt, double start):
        self.phys.kappa = params.get('kappa', 4.0)
        self.phys.theta = params.get('theta', 0.04)
        self.phys.sigma_v = params.get('sigma_v', 0.5)
        self.phys.lambda_j = 0.5
        self.phys.mu_j = -0.05
        self.phys.sigma_j = 0.05
        
        self.dt_base = dt
        # 3000 particles total
        self.particles = <Particle*> malloc(3000 * sizeof(Particle))
        init_imm(&self.phys, self.particles, start)
        
    def __dealloc__(self):
        if self.particles: free(self.particles)
            
    def update(self, double o, double h, double l, double c, double vr, double df):
        cdef IMMState out
        with nogil:
            update_imm(self.particles, &self.phys, o, h, l, c, vr, df, self.dt_base, &out)
            
        return {
            "probs": [out.prob_bull, out.prob_bear, out.prob_neutral],
            "entropy": out.entropy
        }