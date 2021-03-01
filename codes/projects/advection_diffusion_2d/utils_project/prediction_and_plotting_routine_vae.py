#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:17:53 2019

@author: hwan
"""
import sys
sys.path.append('../../../../..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off

# Import src code
from utils_data.data_handler import DataHandler
from neural_networks.nn_vae import VAE
from utils_misc.positivity_constraints import positivity_constraint_log_exp

# Import FEM Code
from Finite_Element_Method.src.load_mesh import load_mesh
from utils_project.plot_fem_function import plot_fem_function

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                              Plot Predictions                               #
###############################################################################
def predict_and_plot(hyperp, options, filepaths):

    #=== Load Observation Indices ===#
    if options.obs_type == 'full':
        obs_dimensions = options.parameter_dimensions
    if options.obs_type == 'obs':
        obs_dimensions = options.num_obs_points

    #=== Data and Latent Dimensions of Autoencoder ===#
    input_dimensions = obs_dimensions
    latent_dimensions = options.parameter_dimensions

    #=== Prepare Data ===#
    data = DataHandler(hyperp, options, filepaths,
                       options.parameter_dimensions, obs_dimensions)
    data.load_data_test()
    if options.add_noise == 1:
        data.add_noise_output_test()
    parameter_test = data.input_test
    state_obs_test = data.output_test

    #=== Load Trained Neural Network ===#
    NN = VAE(hyperp, options,
             input_dimensions, latent_dimensions,
             None, None,
             positivity_constraint_log_exp)
    NN.load_weights(filepaths.trained_NN)

    #=== Selecting Samples ===#
    sample_number = 1
    parameter_test_sample = np.expand_dims(parameter_test[sample_number,:], 0)
    state_obs_test_sample = np.expand_dims(state_obs_test[sample_number,:], 0)

    #=== Predictions ===#
    posterior_mean_pred, posterior_cov_pred = NN.encoder(state_obs_test_sample)
    posterior_pred_draw = NN.reparameterize(posterior_mean_pred, posterior_cov_pred)
    posterior_mean_pred = posterior_mean_pred.numpy().flatten()
    posterior_pred_draw = posterior_pred_draw.numpy().flatten()

    if options.model_aware == 1:
        state_obs_pred_draw = NN.decoder(np.expand_dims(posterior_pred_draw, 0))
        state_obs_pred_draw = state_obs_pred_draw.numpy().flatten()

    #=== Plotting Prediction ===#
    print('================================')
    print('      Plotting Predictions      ')
    print('================================')
    #=== Load Mesh ===#
    nodes, elements, _, _, _, _, _, _ = load_mesh(filepaths.project)

    #=== Plot FEM Functions ===#
    plot_fem_function(filepaths.figure_parameter_test,
                     'True Parameter', 6.0,
                      nodes, elements,
                      parameter_test_sample)
    plot_fem_function(filepaths.figure_parameter_pred,
                      'Parameter Prediction', 6.0,
                      nodes, elements,
                      posterior_pred_draw)
    plot_fem_function(filepaths.figure_posterior_mean,
                      'Parameter Prediction', 6.0,
                      nodes, elements,
                      posterior_mean_pred)
    if options.obs_type == 'full':
        plot_fem_function(filepaths.figure_state_test,
                          'True State', 2.6,
                          nodes, elements,
                          state_obs_test_sample)
        plot_fem_function(filepaths.figure_state_pred,
                          'State Prediction', 2.6,
                          nodes, elements,
                          state_obs_pred_draw)

    print('Predictions plotted')
