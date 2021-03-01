import tensorflow as tf
import pandas as pd
import time

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class SolveFEMAdvectionDiffusion2D:
    def __init__(self, options, filepaths,
                 obs_indices,
                 fem_operator_spatial,
                 fem_operator_implicit_ts, fem_operator_implicit_ts_rhs):

        #=== Defining Attributes ===#
        self.options = options
        self.filepaths = filepaths
        self.obs_indices = tf.cast(obs_indices.flatten(), tf.int32)
        self.fem_operator_spatial = fem_operator_spatial
        self.fem_operator_implicit_ts = fem_operator_implicit_ts
        self.fem_operator_implicit_ts_rhs = fem_operator_implicit_ts_rhs

###############################################################################
#                                 PDE Solvers                                 #
###############################################################################
    def solve_pde(self, parameters):

        ##########################
        #    First Data Point    #
        ##########################
        #=== Initial Structures ===#
        state_current = tf.expand_dims(parameters[0,:],axis=0)
        state_obs = tf.gather(state_current, self.obs_indices, axis=1)

        #=== Time Stepping ===#
        for time_step in range(1, self.options.num_time_steps):
            if self.options.time_stepping_implicit == True:
                state_current = self.time_stepping_implicit(
                        self.fem_operator_implicit_ts, self.fem_operator_implicit_ts_rhs,
                        tf.transpose(state_current))
            if self.options.time_stepping_erk4 == True:
                state_current = self.time_stepping_erk4(
                        -self.fem_operator_spatial, tf.transpose(state_current))
            state_current = tf.transpose(state_current)
            state_obs = tf.concat(
                            [state_obs, tf.gather(
                                state_current, self.obs_indices,
                                axis=1)],
                            axis=1)

        ###############################
        #    Remaining Data Points    #
        ###############################
        for m in range(1, parameters.shape[0]):
            #=== Setting up Initial Structures ===#
            state_current = tf.expand_dims(parameters[0,:],axis=0)
            state_obs_m = tf.gather(state_current, self.obs_indices, axis=1)

            #=== Time Stepping ===#
            for time_step in range(1, self.options.num_time_steps):
                if self.options.time_stepping_implicit == True:
                    state_current = self.time_stepping_implicit(
                            self.fem_operator_implicit_ts, self.fem_operator_implicit_ts_rhs,
                            tf.transpose(state_current))
                if self.options.time_stepping_erk4 == True:
                    state_current = self.time_stepping_erk4(
                            -self.fem_operator_spatial, tf.transpose(state_current))
                state_current = tf.transpose(state_current)
                state_obs_m = tf.concat(
                                [state_obs_m, tf.gather(
                                    state_current, self.obs_indices,
                                    axis=1)],
                                axis=1)
            state_obs = tf.concat([state_obs, state_obs_m], axis=0)

        return state_obs

###############################################################################
#                                 Time Stepping                               #
###############################################################################
    def time_stepping_implicit(self, operator_lhs, operator_rhs, state_current):

        state_current = tf.linalg.solve(
                operator_lhs, tf.linalg.matmul(operator_rhs, state_current))

        return state_current

    def time_stepping_erk4(self, operator, state_current):
        dt = self.options.time_dt

        k_1 = tf.matmul(operator, state_current)
        k_2 = tf.matmul(operator, state_current + (1/2)*dt*k_1)
        k_3 = tf.matmul(operator, state_current + (1/2)*dt*k_2)
        k_4 = tf.matmul(operator, state_current + dt*k_3)

        state_current += (1/6)*dt*(k_1 + 2*k_2 + 2*k_3 + k_4)

        return state_current
