# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "svcj.h":
    int MAX_STATES
    ctypedef struct HMMModel:
        int n_states
        double initial_probs[MAX_STATES]
        double transitions[MAX_STATES][MAX_STATES]
        double means[MAX_STATES]
        double variances[MAX_STATES]
    ctypedef struct HMMResult:
        HMMModel model
        int* viterbi_path
    
    void train_hmm(double* ohlcv, int n_obs, int n_states, double dt, HMMResult* result) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def discover_regimes(object ohlcv, double dt, int n_states=3):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n_obs = data.shape[0]
    
    cdef HMMResult res
    
    with nogil:
        train_hmm(&data[0,0], n_obs, n_states, dt, &res)
        
    model = res.model
    init = np.array(model.initial_probs[:n_states])
    trans = np.array(model.transitions)[:n_states, :n_states]
    means = np.array(model.means[:n_states])
    variances = np.array(model.variances[:n_states])
    
    path = np.array([res.viterbi_path[i] for i in range(n_obs - 1)])
    
    free(res.viterbi_path)
    
    return {
        "initial_probs": init,
        "transitions": trans,
        "means": means,
        "variances": variances,
        "path": path
    }