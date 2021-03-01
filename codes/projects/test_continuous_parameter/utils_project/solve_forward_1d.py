import numpy as np
import pandas as pd
import tensorflow as tf

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class SolveForward1D:
    def __init__(self, options, filepaths, forward_operator, obs_indices):
        #=== Defining Attributes ===#
        self.options = options
        self.filepaths = filepaths
        self.obs_indices = tf.cast(obs_indices, tf.int32)
        self.mesh = np.linspace(0, 1, options.mesh_dimensions, endpoint = True)
        self.forward_operator = forward_operator

###############################################################################
#                               Continuous Linear                             #
###############################################################################
    def continuous_linear(self, parameters):
        #=== Form State Batch ===#
        state = tf.expand_dims(
                tf.tensordot(self.forward_operator, tf.transpose(parameters[0,:]), axes=1),
                axis=0)
        for m in range(1, parameters.shape[0]):
            solution = tf.expand_dims(
                       tf.tensordot(self.forward_operator, tf.transpose(parameters[m,:]), axes=1),
                       axis=0)
            state = tf.concat([state, solution], axis=0)

        #=== Generate Measurement Data ===#
        if self.options.obs_type == 'obs':
            state_obs = tf.gather(state, self.obs_indices, axis=1)
            return tf.squeeze(state_obs)
        else:
            return state
