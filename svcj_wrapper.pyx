# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cdef extern from "svcj.h":
    ctypedef struct HMMModel:
        pass
    
    void compute_log_returns(double* ohlcv, int n, double* out) nogil
    void run_baum_welch(double* returns, int n, HMMModel* model) nogil
    void decode_states_viterbi(double* returns, int n, HMMModel* model, int* out) nogil
    
cdef np.ndarray[double, ndim=2, mode='c'] _sanitize(object d):
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))

def fit_hmm(object ohlcv):
    """
    Fits the Hidden Markov Model to OHLCV data.
    Returns: The discovered Physics (Model Parameters).
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    cdef int n_ret = n - 1
    
    cdef np.ndarray[double, ndim=1] returns = np.zeros(n_ret)
    compute_log_returns(&data[0,0], n, &returns[0])
    
    # Create Model struct in C memory
    cdef HMMModel model
    
    # Run Solver
    run_baum_welch(&returns[0], n_ret, &model)
    
    # Unpack to Python dict
    return {
        "states": [
            {'mu': model.states[0].mu, 'sigma': model.states[0].sigma},
            {'mu': model.states[1].mu, 'sigma': model.states[1].sigma},
            {'mu': model.states[2].mu, 'sigma': model.states[2].sigma}
        ],
        "transitions": np.asarray(<double[:3,:3]> model.transitions)
    }

def decode_regime_path(object ohlcv, dict model_params):
    """
    Decodes the most likely sequence of hidden states.
    Returns: A 1D array of state labels (0, 1, or 2).
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize(ohlcv)
    cdef int n = data.shape[0]
    cdef int n_ret = n - 1
    
    cdef np.ndarray[double, ndim=1] returns = np.zeros(n_ret)
    compute_log_returns(&data[0,0], n, &returns[0])
    
    # Pack Python dict -> C struct
    cdef HMMModel model
    for i in range(3):
        model.states[i].mu = model_params['states'][i]['mu']
        model.states[i].sigma = model_params['states'][i]['sigma']
    
    cdef np.ndarray[double, ndim=2, mode='c'] trans = np.asarray(model_params['transitions'])
    for i in range(3):
        for j in range(3):
            model.transitions[i][j] = trans[i,j]
    
    # Allocate output buffer
    cdef np.ndarray[int, ndim=1, mode='c'] path = np.zeros(n_ret, dtype=np.int32)
    
    # Run Viterbi
    decode_states_viterbi(&returns[0], n_ret, &model, &path[0])
    
    return path