#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:45:14 2020

@author: hwan
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.initializers import RandomNormal
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                           Variational Autoencoder                           #
###############################################################################
class VAE(tf.keras.Model):
    def __init__(self, hyperp, options,
                 input_dimensions, latent_dimensions,
                 kernel_initializer, bias_initializer,
                 positivity_constraint):
        super(VAE, self).__init__()

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

        #=== Encoder and Decoder ===#
        self.encoder = Encoder(options,
                               hyperp.num_hidden_layers_encoder + 1,
                               self.architecture, self.activations,
                               kernel_initializer, bias_initializer)
        if self.options.model_aware == 1:
            self.decoder = Decoder(options,
                                   hyperp.num_hidden_layers_encoder + 1,
                                   self.architecture, self.activations,
                                   kernel_initializer, bias_initializer,
                                   len(self.architecture) - 1)

    #=== Variational Autoencoder Propagation ===#
    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=mean.shape)
        return self.positivity_constraint(mean + eps*tf.exp(log_var*0.5))

    def call(self, X):
        post_mean, log_post_var = self.encoder(X)
        if self.options.model_augmented == True:
            return post_mean, log_post_var
        if self.options.model_aware == True:
            z = self.reparameterize(post_mean, log_post_var)
            likelihood_mean = self.decoder(z)
            return likelihood_mean

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
        self.hidden_layers_encoder = [] # This will be a list of layers

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
        self.hidden_layers_decoder = [] # This will be a list of layers

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
