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
    ctypedef struct ActionProfile:
        int direction, horizon_bars
        double stop_sigma, target_sigma
    ctypedef struct SimulationResult:
        double ev, std_dev, win_rate, t_stat, kelly_q
        
    void generate_prior_swarm(double* ohlcv, int n, double dt, Particle* swarm) nogil
    void run_particle_filter_step(Particle* swarm, double ret, double dt, FilterStats* stats) nogil
    void run_ev_simulation(Particle* swarm, double price, double dt, ActionProfile act, SimulationResult* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

cdef class ParticleFilter:
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
            
    def update(self, double price_now, double price_prev):
        cdef double ret = np.log(price_now / price_prev)
        cdef FilterStats stats
        run_particle_filter_step(self.swarm, ret, self.dt, &stats)
        return {
            "spot_vol": stats.spot_vol_mean,
            "z_score": stats.innovation_z,
            "ess": stats.ess
        }
        
    def simulate(self, double current_price, int direction, double stop_sigma, double target_sigma, int horizon):
        """
        Runs Monte Carlo on current swarm state to value an action.
        """
        cdef ActionProfile act
        act.direction = direction
        act.stop_sigma = stop_sigma
        act.target_sigma = target_sigma
        act.horizon_bars = horizon
        
        cdef SimulationResult res
        
        # Release GIL for Simulation
        with nogil:
            run_ev_simulation(self.swarm, current_price, self.dt, act, &res)
            
        return {
            "ev": res.ev,
            "std_dev": res.std_dev,
            "win_rate": res.win_rate,
            "t_stat": res.t_stat,
            "kelly": res.kelly_q
        }