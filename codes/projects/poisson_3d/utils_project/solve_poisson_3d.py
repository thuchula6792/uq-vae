import tensorflow as tf
import numpy as np
import pandas as pd
import time

from Utilities.integrals_pwl_prestiffness import integrals_pwl_prestiffness

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 Standard                                    #
###############################################################################
def solve_PDE_standard(options, filepaths,
                       parameters,
                       obs_indices, nodes, elements,
                       boundary_matrix, load_vector):

    #=== Create Storage ===#
    stiffness_matrix = np.zeros(
            (options.parameter_dimensions, options.parameter_dimensions))

    #=== Construct Matrices ===#
    for k in range(0, elements.shape[0]):
        ver = elements[k,:]
        vertices_coords = nodes[ver,:]
        p_k = parameters.numpy()[ver]
        stiffness_matrix[np.ix_(ver,ver)] +=\
                integrals_pwl_prestiffness(vertices_coords)*(p_k[0] + p_k[1] + p_k[2])

    return np.transpose(np.linalg.solve(stiffness_matrix + boundary_matrix, load_vector))

###############################################################################
#                                 Sensitivity                                 #
###############################################################################
def construct_sensitivity(options, filepaths,
                          parameters, state,
                          obs_indices, nodes, elements,
                          boundary_matrix, load_vector):

    #=== Create Storage ===#
    stiffness_matrix = np.zeros(
            (options.parameter_dimensions, options.parameter_dimensions))
    partial_derivative_parameter = np.zeros(
            (options.parameter_dimensions, options.parameter_dimensions))
    partial_derivative_matrix = np.zeros(
            (options.parameter_dimensions, options.parameter_dimensions))

    #=== Construct Matrices ===#
    for k in range(0, elements.shape[0]):
        ver = elements[k,:]
        vertices_coords = nodes[ver,:]
        p_k = parameters.numpy()[ver]
        stiffness_matrix[np.ix_(ver,ver)] +=\
                integrals_pwl_prestiffness(vertices_coords)*(p_k[0] + p_k[1] + p_k[2])
        partial_derivative_parameter[np.ix_(ver,ver)] +=\
                integrals_pwl_prestiffness(vertices_coords)

    #=== Forming Objects ===#
    partial_derivative_matrix = np.repeat(
            np.matmul(
                partial_derivative_parameter, np.expand_dims(state, axis = 1)),
            repeats=options.parameter_dimensions, axis=1)

    return -np.linalg.solve(stiffness_matrix + boundary_matrix, partial_derivative_matrix)
