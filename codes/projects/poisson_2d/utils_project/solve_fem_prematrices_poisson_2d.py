import tensorflow as tf
import numpy as np
import pandas as pd
import time

from utils_project.integrals_pwl_prestiffness import integrals_pwl_prestiffness

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class SolveFEMPrematricesPoisson2D:
    def __init__(self, options, filepaths,
                 obs_indices,
                 prestiffness,
                 boundary_matrix, load_vector):

        #=== Defining Attributes ===#
        self.options = options
        self.filepaths = filepaths
        self.obs_indices = tf.cast(obs_indices, tf.int32)
        self.prestiffness = prestiffness
        self.boundary_matrix = boundary_matrix
        self.load_vector = load_vector

###############################################################################
#                                 PDE Solvers                                 #
###############################################################################
    def solve_pde_prematrices(self, parameters):

        #=== Solving PDE ===#
        prestiffness = tf.linalg.matmul(self.prestiffness, tf.transpose(parameters))
        stiffness_matrix = tf.reshape(prestiffness[:,0],
                    (self.options.parameter_dimensions, self.options.parameter_dimensions))
        state = tf.transpose(
                tf.linalg.solve(stiffness_matrix + self.boundary_matrix, self.load_vector))
        for n in range(1, parameters.shape[0]):
            stiffness_matrix = tf.reshape(self.prestiffness[:,n],
                    (self.options.parameter_dimensions, self.options.parameter_dimensions))
            solution = tf.linalg.solve(stiffness_matrix + self.boundary_matrix, self.load_vector)
            state = tf.concat([state, tf.transpose(solution)], axis=0)

        #=== Generate Measurement Data ===#
        if options.obs_type == 'obs':
            state_obs = tf.gather(state, self.obs_indices, axis=1)
            return tf.squeeze(state_obs)
        else:
            return state

    def solve_pde_prematrices(self, parameters):

        state = tf.Variable(tf.zeros((parameters.shape[0], self.options.parameter_dimensions)))

        #=== Solving PDE ===#
        prestiffness = tf.linalg.matmul(self.prestiffness, tf.transpose(parameters))
        for n in range(0, parameters.shape[0]):
            stiffness_matrix = tf.reshape(prestiffness[:,n:n+1],
                    (self.options.parameter_dimensions, self.options.parameter_dimensions))
            state[n:n+1,:].assign(tf.transpose(
                    tf.linalg.solve(stiffness_matrix + self.boundary_matrix, self.load_vector)))

        #=== Generate Measurement Data ===#
        if self.options.obs_type == 'obs':
            state = state[:,self.obs_indices]

        return state

###############################################################################
#                           Using Sparse Prematrices                          #
###############################################################################
    def solve_pde_prematrices_sparse(self, parameters):

        #=== Solving PDE ===#
        stiffness_matrix = tf.reshape(
                tf.sparse.sparse_dense_matmul(
                    self.prestiffness, tf.expand_dims(tf.transpose(parameters[0,:]), axis=1)),
                    (self.options.parameter_dimensions, self.options.parameter_dimensions))
        state = tf.transpose(
                tf.linalg.solve(stiffness_matrix + self.boundary_matrix, self.load_vector))
        for n in range(1, parameters.shape[0]):
            stiffness_matrix = tf.reshape(
                    tf.sparse.sparse_dense_matmul(
                        self.prestiffness, tf.expand_dims(tf.transpose(parameters[n,:]), axis=1)),
                    (self.options.parameter_dimensions, self.options.parameter_dimensions))
            solution = tf.linalg.solve(stiffness_matrix + self.boundary_matrix, self.load_vector)
            state = tf.concat([state, tf.transpose(solution)], axis=0)

        #=== Generate Measurement Data ===#
        if self.options.obs_type == 'obs':
            state_obs = tf.gather(state, self.obs_indices, axis=1)
            return tf.squeeze(state_obs)
        else:
            return tf.squeeze(state)
