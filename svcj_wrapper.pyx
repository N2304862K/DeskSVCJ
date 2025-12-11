# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    ctypedef struct Particle:
        double kappa, theta, sigma_v, rho, lambda_j, v, weight
    ctypedef struct FilterStats:
        double spot_vol_mean, spot_vol_std, innovation_z, entropy, ess
    ctypedef struct MarketMicrostructure:
        double spread_bps, impact_coef, stop_sigma, target_sigma_init, decay_rate
        int horizon
    ctypedef struct ContrastiveResult:
        double alpha_long, alpha_short, t_stat_long, t_stat_short
        double cohens_d, friction_cost_avg
        
    void generate_prior_swarm(double* ohlcv, int n, double dt, Particle* out) nogil
    void run_particle_filter_step(Particle* sw, double ret, double dt, FilterStats* out) nogil
    void run_contrastive_simulation(Particle* sw, double p, double dt, MarketMicrostructure m, ContrastiveResult* r) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

cdef class ParticleEngine:
    cdef Particle* swarm
    cdef public double dt
    cdef int swarm_size
    
    def __cinit__(self, object ohlcv, double dt):
        cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
        self.swarm_size = 5000
        self.swarm = <Particle*> malloc(self.swarm_size * sizeof(Particle))
        self.dt = dt
        generate_prior_swarm(&data[0,0], data.shape[0], dt, self.swarm)
        
    def __dealloc__(self):
        if self.swarm is not NULL: free(self.swarm)
            
    def update(self, double p_now, double p_prev):
        cdef double ret = np.log(p_now / p_prev)
        cdef FilterStats s
        run_particle_filter_step(self.swarm, ret, self.dt, &s)
        return {
            "spot_vol": s.spot_vol_mean,
            "z": s.innovation_z,
            "ess": s.ess,
            "entropy": s.entropy
        }
    
    def evaluate_contrastive(self, double price, dict config):
        """
        Runs the full Long vs Short vs Hold tournament.
        """
        cdef MarketMicrostructure m
        m.spread_bps = config.get('spread_bps', 0.0002)
        m.impact_coef = config.get('impact', 0.1)
        m.stop_sigma = config.get('stop', 2.0)
        m.target_sigma_init = config.get('target', 4.0)
        m.decay_rate = config.get('decay', 0.05)
        m.horizon = config.get('horizon', 12)
        
        cdef ContrastiveResult r
        
        # Release GIL for MC Simulation
        with nogil:
            run_contrastive_simulation(self.swarm, price, self.dt, m, &r)
            
        return {
            "alpha_long": r.alpha_long,
            "alpha_short": r.alpha_short,
            "t_long": r.t_stat_long,
            "t_short": r.t_stat_short,
            "separation": r.cohens_d,
            "friction": r.friction_cost_avg
        }