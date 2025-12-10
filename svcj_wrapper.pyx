# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer

cdef extern from "svcj.h":
    ctypedef struct EvolvingSystemState:
        pass
    ctypedef struct InstantMetrics:
        double expected_return, expected_vol, escape_velocity, swarm_entropy
        
    EvolvingSystemState* initialize_system(double* ohlcv, int n, double dt, int n_p) nogil
    void run_system_step(EvolvingSystemState* state, double ret, double vol, InstantMetrics* out) nogil
    void cleanup_system(EvolvingSystemState* state) nogil

# --- The Stateless Interface (Correct & Safe) ---

# This is the C function that will be called by the PyCapsule destructor
cdef void capsule_cleanup(object capsule):
    # Get the C pointer from the capsule and call our cleanup function
    cdef EvolvingSystemState* state_ptr = <EvolvingSystemState*>PyCapsule_GetPointer(capsule, "EvolvingSystemState")
    if state_ptr is not NULL:
        cleanup_system(state_ptr)

def initialize_engine(object ohlcv, double dt, int num_particles=1500):
    """
    Initializes the C-Core state and returns a safe Python handle (PyCapsule).
    """
    cdef np.ndarray[double, ndim=2, mode='c'] data = np.ascontiguousarray(ohlcv, dtype=np.float64)
    cdef int n = data.shape[0]
    
    # Call the C initializer
    cdef EvolvingSystemState* state_ptr
    with nogil:
        state_ptr = initialize_system(&data[0,0], n, dt, num_particles)
        
    if state_ptr is NULL:
        raise MemoryError("Failed to initialize C-Core state.")
        
    # Create a PyCapsule to safely manage the pointer in Python
    # It stores the pointer and a destructor function
    return PyCapsule_New(state_ptr, "EvolvingSystemState", capsule_cleanup)

def run_engine_step(object capsule, double new_ret, double new_vol):
    """
    Takes the Python handle, runs one step in the C-Core, returns metrics.
    """
    cdef EvolvingSystemState* state_ptr = <EvolvingSystemState*>PyCapsule_GetPointer(capsule, "EvolvingSystemState")
    if state_ptr is NULL:
        raise ValueError("Invalid Engine Handle.")
        
    cdef InstantMetrics metrics
    
    # Call the C update function
    with nogil:
        run_system_step(state_ptr, new_ret, new_vol, &metrics)
    
    return {
        "ev_ret": metrics.expected_return,
        "ev_vol": metrics.expected_vol,
        "escape": metrics.escape_velocity,
        "entropy": metrics.swarm_entropy
    }