# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    ctypedef struct Particle:
        double mu, kappa, theta, sigma_v, rho, lambda_j
        double weight
    ctypedef struct GravityDistribution:
        double mean[6], cov[36]
    ctypedef struct InstantState:
        double expected_return, expected_vol, mahalanobis_dist, kl_divergence, swarm_entropy
    
    void run_gravity_scan(double* ohlcv, int len, double dt, GravityDistribution* out) nogil
    void generate_prior_swarm(GravityDistribution* anchor, int n, Particle* out) nogil
    void run_particle_filter_step(Particle* curr, int n, double r, double v, double av, double dt, GravityDistribution* anchor, Particle* next, InstantState* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

# The Main Class: Evolving Particle Filter
cdef class EvolvingFilter:
    cdef GravityDistribution anchor
    cdef Particle* swarm
    cdef int n_particles
    cdef double dt
    cdef double avg_vol
    
    def __init__(self, object ohlcv, double dt, int num_particles=1000):
        cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
        
        # 1. Establish Gravity
        run_gravity_scan(&data[0,0], data.shape[0], dt, &self.anchor)
        
        # 2. Seed the Swarm
        self.n_particles = num_particles
        self.dt = dt
        self.swarm = <Particle*> malloc(num_particles * sizeof(Particle))
        generate_prior_swarm(&self.anchor, num_particles, self.swarm)
        
        # Set initial avg volume
        self.avg_vol = np.mean(ohlcv[:, 4])
        
    def __dealloc__(self):
        free(self.swarm)
        
    def update(self, double new_ret, double new_vol):
        cdef Particle* next_gen = <Particle*> malloc(self.n_particles * sizeof(Particle))
        cdef InstantState state
        
        run_particle_filter_step(
            self.swarm, self.n_particles, new_ret, new_vol, self.avg_vol,
            self.dt, &self.anchor, next_gen, &state
        )
        
        # Swap pointers for next iteration
        free(self.swarm)
        self.swarm = next_gen
        
        return {
            "ev_ret": state.expected_return,
            "ev_vol": state.expected_vol,
            "escape_dist": state.mahalanobis_dist,
            "surprise": state.kl_divergence,
            "entropy": state.swarm_entropy
        }