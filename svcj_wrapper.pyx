# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj_swarm.h":
    # Structs must match C header exactly
    ctypedef struct Particle:
        double v, mu, rho
        int jump_state
        double weight
        
    ctypedef struct SwarmParams:
        double kappa, theta, sigma_v, rho_mean, lambda_j, mu_j, sigma_j
        
    ctypedef struct SwarmMetrics:
        double expected_return, risk_volatility, swarm_entropy, regime_prob, effective_rho
    
    void init_swarm(Particle* swarm, SwarmParams* p) nogil
    void predict_step(Particle* swarm, SwarmParams* p, double dt, double diurnal) nogil
    void update_step(Particle* swarm, SwarmParams* p, double ret, double rng, double dt) nogil
    void resample_regularized(Particle* swarm, SwarmParams* p) nogil
    void calc_swarm_metrics(Particle* swarm, SwarmMetrics* out) nogil

cdef class SwarmEngine:
    cdef Particle* particles
    cdef SwarmParams params
    cdef double dt
    cdef int n_particles
    
    def __init__(self, dict physics, double dt):
        self.n_particles = 2000
        self.particles = <Particle*> malloc(self.n_particles * sizeof(Particle))
        self.dt = dt
        
        # Load Physics
        self.params.kappa = physics['kappa']
        self.params.theta = physics['theta']
        self.params.sigma_v = physics['sigma_v']
        self.params.rho_mean = physics['rho']
        self.params.lambda_j = physics['lambda_j']
        self.params.mu_j = -0.05 # Default downside skew for jumps
        self.params.sigma_j = 0.05
        
        # Init
        with nogil:
            init_swarm(self.particles, &self.params)
            
    def __dealloc__(self):
        free(self.particles)
        
    def update(self, double ret, double range_sq, double diurnal_factor):
        """
        Steps the Swarm forward 1 tick.
        ret: Log Return (Close-to-Close)
        range_sq: (High-Low)^2 / Close^2 (Parkinson Vol Proxy)
        diurnal_factor: 1.0 = Average, 2.0 = Open/Close, 0.5 = Lunch
        """
        cdef SwarmMetrics m
        
        with nogil:
            # 1. Predict (Drift & Diffuse)
            predict_step(self.particles, &self.params, self.dt, diurnal_factor)
            
            # 2. Update (Weight by Evidence)
            update_step(self.particles, &self.params, ret, range_sq, self.dt)
            
            # 3. Resample (Survival of fittest)
            resample_regularized(self.particles, &self.params)
            
            # 4. Measure
            calc_swarm_metrics(self.particles, &m)
            
        return {
            "ev_drift": m.expected_return,
            "risk_vol": m.risk_volatility,
            "entropy": m.swarm_entropy,
            "jump_prob": m.regime_prob,
            "rho": m.effective_rho
        }