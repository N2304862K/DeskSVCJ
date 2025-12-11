# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    ctypedef struct Particle:
        double v, mu, weight
        int regime
    ctypedef struct IMMState:
        double p_bull, p_bear, p_neutral, entropy
        double kappa, theta, sigma_v, dt

    void init_particle_set(Particle* particles, double theta, double start_price) nogil
    void update_particles(Particle* particles, IMMState* state, double o, double h, double l, double c, double vf, double rf) nogil
    
    int N_PARTICLES

# The Stateful Class
cdef class IMMFilter:
    cdef Particle* particles
    cdef IMMState state
    
    def __cinit__(self, dict physics, double dt, double start_price):
        # Allocate memory for the particle swarm
        self.particles = <Particle*> malloc(N_PARTICLES * sizeof(Particle))
        if not self.particles:
            raise MemoryError()
            
        # Set Physics
        self.state.theta = physics['theta']
        self.state.kappa = physics['kappa']
        self.state.sigma_v = physics['sigma_v']
        self.state.dt = dt
        
        # Initialize
        init_particle_set(self.particles, self.state.theta, start_price)
        
    def __dealloc__(self):
        # Clean up C memory
        if self.particles:
            free(self.particles)
            
    def update(self, double o, double h, double l, double c, double vol_factor, double range_factor):
        # This function runs in microseconds. It just calls the C core.
        update_particles(self.particles, &self.state, o, h, l, c, vol_factor, range_factor)
        
        return {
            "probs": [self.state.p_bull, self.state.p_bear, self.state.p_neutral],
            "entropy": self.state.entropy
        }