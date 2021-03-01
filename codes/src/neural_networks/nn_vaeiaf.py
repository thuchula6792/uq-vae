#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:45:14 2020

@author: hwan
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.initializers import RandomNormal
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                           Variational Autoencoder                           #
###############################################################################
class VAEIAF(tf.keras.Model):
    def __init__(self, hyperp, options,
                 input_dimensions, latent_dimensions,
                 kernel_initializer, bias_initializer,
                 kernel_initializer_iaf, bias_initializer_iaf,
                 positivity_constraint):
        super(VAEIAF, self).__init__()

        #=== Define Architecture and Create Layer Storage ===#
        self.architecture = [input_dimensions] +\
                [hyperp.num_hidden_nodes_encoder]*hyperp.num_hidden_layers_encoder +\
                [latent_dimensions + latent_dimensions] +\
                [hyperp.num_hidden_nodes_decoder]*hyperp.num_hidden_layers_decoder +\
                [input_dimensions]

        #=== Define Other Attributes ===#
        self.options = options
        self.positivity_constraint = positivity_constraint
        self.activations = ['not required'] +\
                [hyperp.activation]*hyperp.num_hidden_layers_encoder +\
                ['linear'] +\
                [hyperp.activation]*hyperp.num_hidden_layers_decoder +\
                ['linear']

        #=== Encoder, IAF Chain and Decoder ===#
        self.encoder = Encoder(options,
                               hyperp.num_hidden_layers_encoder + 1,
                               self.architecture, self.activations,
                               kernel_initializer, bias_initializer)
        self.iaf_chain_encoder = IAFChainEncoder(options.iaf_lstm_update,
                                                 hyperp.num_iaf_transforms,
                                                 hyperp.num_hidden_nodes_iaf,
                                                 hyperp.activation_iaf,
                                                 kernel_initializer_iaf, bias_initializer_iaf)
        if self.options.model_aware == True:
            self.decoder = Decoder(options,
                                   hyperp.num_hidden_layers_encoder + 1,
                                   self.architecture, self.activations,
                                   kernel_initializer, bias_initializer,
                                   len(self.architecture) - 1)

    #=== Variational Autoencoder Propagation ===#
    def reparameterize(self, mean, log_var):
        return self.iaf_chain_encoder((mean, log_var),
                                        sample_flag = True, infer_flag = False)

    def call(self, X):
        post_mean, log_post_var = self.encoder(X)
        if self.options.model_augmented == True:
            return self.positivity_constraint(reparameterize(post_mean, log_post_var))
        if self.options.model_aware == True:
            z = self.reparameterize(post_mean, log_post_var)
            return self.decoder(self.positivity_constraint(z))

###############################################################################
#                                  Encoder                                    #
###############################################################################
class Encoder(tf.keras.layers.Layer):
    def __init__(self, options,
                 truncation_layer,
                 architecture,
                 activations,
                 kernel_initializer, bias_initializer):
        super(Encoder, self).__init__()

        self.options = options
        self.truncation_layer = truncation_layer
        self.hidden_layers_encoder = []

        for l in range(1, truncation_layer+1):
            hidden_layer_encoder = tf.keras.layers.Dense(units = architecture[l],
                                                         activation = activations[l],
                                                         use_bias = True,
                                                         kernel_initializer = kernel_initializer,
                                                         bias_initializer = bias_initializer,
                                                         name = "W" + str(l))
            self.hidden_layers_encoder.append(hidden_layer_encoder)

    def call(self, X):
        for hidden_layer in enumerate(self.hidden_layers_encoder):
            if self.options.resnet == True\
                    and 0 < hidden_layer[0] < self.truncation_layer-1:
                X += hidden_layer[1](X)
            else:
                X = hidden_layer[1](X)
        post_mean, log_post_var = tf.split(X, num_or_size_splits=2, axis=1)
        return post_mean, log_post_var

###############################################################################
#                                  Decoder                                    #
###############################################################################
class Decoder(tf.keras.layers.Layer):
    def __init__(self, options,
                 truncation_layer,
                 architecture,
                 activations,
                 kernel_initializer, bias_initializer,
                 last_layer_index):
        super(Decoder, self).__init__()

        self.options = options
        self.truncation_layer = truncation_layer
        self.last_layer_index = last_layer_index
        self.hidden_layers_decoder = []

        for l in range(truncation_layer+1, last_layer_index+1):
            hidden_layer_decoder = tf.keras.layers.Dense(units = architecture[l],
                                                         activation = activations[l],
                                                         use_bias = True,
                                                         kernel_initializer = kernel_initializer,
                                                         bias_initializer = bias_initializer,
                                                         name = "W" + str(l))
            self.hidden_layers_decoder.append(hidden_layer_decoder)

    def call(self, X):
        for hidden_layer in enumerate(self.hidden_layers_decoder):
            if self.options.resnet == True\
                    and self.truncation_layer < hidden_layer[0]+self.truncation_layer\
                            < self.last_layer_index-1:
                X += hidden_layer[1](X)
            else:
                X = hidden_layer[1](X)
        return X

###############################################################################
#                    Chain of Inverse Autoregressive Flow                     #
###############################################################################
class IAFChainEncoder(tf.keras.layers.Layer):
    def __init__(self, iaf_lstm_update_flag,
                 num_iaf_transforms,
                 hidden_units,
                 activation,
                 kernel_initializer, bias_initializer):
        super(IAFChainPosterior, self).__init__()

        #=== Attributes ===#
        self.iaf_lstm_update_flag = iaf_lstm_update_flag
        self.num_iaf_transforms = num_iaf_transforms
        self.hidden_units = hidden_units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        #=== IAF Chain ===#
        latent_dimensions = input_shape[0][1]
        self.event_shape = [latent_dimensions]
        bijectors_list = []
        for i in range(0, self.num_iaf_transforms):
            bijectors_list.append(tfb.Invert(
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn = Made(params=2,
                                                  event_shape = self.event_shape,
                                                  hidden_units = self.hidden_units,
                                                  activation = self.activation,
                                                  kernel_initializer = self.kernel_initializer,
                                                  bias_initializer = self.bias_initializer,
                                                  lstm_flag = self.iaf_lstm_update_flag,
                                                  name = "IAF_W" + str(i)))))
            bijectors_list.append(tfb.Permute(list(reversed(range(latent_dimensions)))))
        self.iaf_chain = tfb.Chain(bijectors_list)

    def call(self, inputs, sample_flag = True, infer_flag = False):
        mean = inputs[0]
        log_var = inputs[1]

        #=== Base Distribution ===#
        base_distribution = tfd.MultivariateNormalDiag(loc = mean,
                                                       scale_diag = tf.exp(0.5*log_var))
        #=== Transformed Distribution ===#
        self.distribution = tfd.TransformedDistribution(distribution = base_distribution,
                                                        bijector = self.iaf_chain)

        #=== Inference and Sampling ===#
        sample_draw = self.distribution.sample()
        self.foo = self.distribution.variables # for some reason,
                                               # this is required to register trainable variables
        if sample_flag == True:
            return sample_draw
        if infer_flag == True:
            return self.distribution.log_prob(sample_draw)

###############################################################################
#                          Masked Autoregressive Flow                         #
###############################################################################
class Made(tf.keras.layers.Layer):
    def __init__(self, params,
                 event_shape,
                 hidden_units,
                 activation,
                 kernel_initializer, bias_initializer,
                 lstm_flag,
                 name):
        super(Made, self).__init__(name = name)

        self.lstm_flag = lstm_flag
        self.network = tfb.AutoregressiveNetwork(params = params,
                                                 event_shape = event_shape,
                                                 hidden_units = [hidden_units, hidden_units],
                                                 activation = activation,
                                                 kernel_initializer = kernel_initializer,
                                                 bias_initializer = bias_initializer)

    def call(self, X):
        mean, log_var = tf.unstack(self.network(X), num=2, axis=-1)
        if self.lstm_flag == False:
            return mean, tf.math.tanh(log_var)
        else:
            s = tf.math.sigmoid(log_var)
            mean = mean - tf.multiply(s, mean)
            return mean, tf.math.log_sigmoid(log_var)
