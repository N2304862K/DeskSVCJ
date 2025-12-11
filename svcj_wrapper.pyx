# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    int N_STATES
    
    ctypedef struct RegimeParams:
        double mu, sigma
    ctypedef struct HMM:
        RegimeParams states[N_STATES]
        double transitions[N_STATES][N_STATES]
        double initial_probs[N_STATES]
    
    void compute_log_returns(double* ohlcv, int n, double* out) nogil
    void train_svcj_hmm(double* returns, int n, double dt, int max_iter, HMM* out) nogil
    void decode_regime_path(double* returns, int n, double dt, HMM* model, int* out) nogil

cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def train(object ohlcv, double dt, int max_iter=100):
    """
    Fits the Hidden Markov Model to the data.
    Returns: The discovered Physics (Params, Transitions).
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    cdef int n_ret = n - 1
    
    cdef np.ndarray[double, ndim=1, mode='c'] ret = np.zeros(n_ret, dtype=np.float64)
    compute_log_returns(&data[0,0], n, &ret[0])
    
    cdef HMM model
    train_svcj_hmm(&ret[0], n_ret, dt, max_iter, &model)
    
    # Unpack to Python dict
    states = []
    for i in range(N_STATES):
        states.append({"mu": model.states[i].mu, "sigma": model.states[i].sigma})
        
    transitions = np.zeros((N_STATES, N_STATES))
    for i in range(N_STATES):
        for j in range(N_STATES):
            transitions[i, j] = model.transitions[i][j]
            
    return {
        "states": states,
        "transitions": transitions,
        "initial_probs": np.array(model.initial_probs)
    }

def decode(object ohlcv, double dt, dict hmm_model):
    """
    Uses a trained HMM to find the most likely regime path.
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    cdef int n_ret = n - 1
    
    cdef np.ndarray[double, ndim=1, mode='c'] ret = np.zeros(n_ret, dtype=np.float64)
    compute_log_returns(&data[0,0], n, &ret[0])
    
    # Pack model from Python dict to C struct
    cdef HMM model
    for i in range(N_STATES):
        s = hmm_model['states'][i]
        model.states[i].mu = s['mu']
        model.states[i].sigma = s['sigma']
        
        model.initial_probs[i] = hmm_model['initial_probs'][i]
        for j in range(N_STATES):
            model.transitions[i][j] = hmm_model['transitions'][i,j]
            
    cdef np.ndarray[int, ndim=1, mode='c'] path = np.zeros(n_ret, dtype=np.int32)
    decode_regime_path(&ret[0], n_ret, dt, &model, &path[0])
    
    return path