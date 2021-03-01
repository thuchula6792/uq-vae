import tensorflow as tf
import numpy as np
import pandas as pd
import time

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 Dirichlet                                   #
###############################################################################
class SolveFEMLinearDirichlet1D:
    def __init__(self, options, filepaths,
                 obs_indices,
                 forward_matrix, mass_matrix):

        #=== Defining Attributes ===#
        self.options = options
        self.filepaths = filepaths
        self.obs_indices = tf.cast(obs_indices, tf.int32)
        self.forward_matrix = forward_matrix
        self.mass_matrix = mass_matrix

        #=== Implementing Dirchlet Boundary Conditions ===#
        self.dirichlet_mult_vec = np.ones([options.parameter_dimensions],np.float32)
        self.dirichlet_mult_vec[0] = 0
        self.dirichlet_mult_vec[-1] = 0
        self.dirichlet_mult_vec = tf.cast(self.dirichlet_mult_vec, tf.float32)
        self.dirichlet_add_vec = np.zeros([options.parameter_dimensions],np.float32)
        self.dirichlet_add_vec[-1] = 1
        self.dirichlet_add_vec = tf.cast(self.dirichlet_add_vec, tf.float32)

    def solve_pde(self, parameters):
        #=== Solving PDE ===#
        rhs = tf.linalg.matmul(
                tf.expand_dims(parameters[0,:], axis=0), tf.transpose(self.mass_matrix))
        rhs = tf.math.multiply(self.dirichlet_mult_vec, rhs)
        rhs = tf.math.add(self.dirichlet_add_vec, rhs)
        state = tf.linalg.matmul(rhs, tf.transpose(self.forward_matrix))
        for n in range(1, parameters.shape[0]):
            rhs = tf.linalg.matmul(
                    tf.expand_dims(parameters[n,:], axis=0), tf.transpose(self.mass_matrix))
            rhs = tf.math.multiply(self.dirichlet_mult_vec, rhs)
            rhs = tf.math.add(self.dirichlet_add_vec, rhs)
            solution = tf.linalg.matmul(rhs, tf.transpose(self.forward_matrix))
            state = tf.concat([state, solution], axis=0)

        #=== Generate Measurement Data ===#
        if self.options.obs_type == 'obs':
            state_obs = tf.gather(state, self.obs_indices, axis=1)
            return tf.squeeze(state_obs)
        else:
            return state

###############################################################################
#                                   Neumann                                   #
###############################################################################
class SolveFEMLinearNeumann:
    def __init__(self, options, filepaths,
                 obs_indices,
                 forward_matrix, mass_matrix):

        #=== Defining Attributes ===#
        self.options = options
        self.filepaths = filepaths
        self.obs_indices = tf.cast(obs_indices, tf.int32)
        self.forward_matrix = forward_matrix
        self.mass_matrix = mass_matrix

    def solve_pde(self, parameters):
        #=== Solving PDE ===#
        rhs = tf.linalg.matmul(
                tf.expand_dims(parameters[0,:], axis=0), tf.transpose(self.mass_matrix))
        state = tf.linalg.matmul(rhs, tf.transpose(self.forward_matrix))
        for n in range(1, parameters.shape[0]):
            rhs = tf.linalg.matmul(
                    tf.expand_dims(parameters[n,:], axis=0), tf.transpose(self.mass_matrix))
            solution = tf.linalg.matmul(rhs, tf.transpose(self.forward_matrix))
            state = tf.concat([state, solution], axis=0)

        #=== Generate Measurement Data ===#
        if self.options.obs_type == 'obs':
            state_obs = tf.gather(state, self.obs_indices, axis=1)
            return tf.squeeze(state_obs)
        else:
            return state
