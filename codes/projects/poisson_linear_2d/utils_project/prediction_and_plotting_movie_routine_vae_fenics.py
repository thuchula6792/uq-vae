#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:17:53 2019

@author: hwan
"""
import sys
import os

import numpy as np
import pandas as pd

# Import src code
from utils_data.data_handler import DataHandler
from neural_networks.nn_vae import VAE
from utils_misc.positivity_constraints import positivity_constraint_exp,\
                                              positivity_constraint_log_exp
from utils_io.make_movie import make_movie

# Import project utilities
sys.path.insert(0, os.path.realpath('../../../../../fenics-simulations/src'))
from utils_project.plot_fem_function_fenics_2d import plot_fem_function_fenics_2d
from utils_project.plot_cross_section import plot_cross_section

# Import FEniCS code
from utils_mesh.construct_mesh_rectangular import construct_mesh

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                              Plot Predictions                               #
###############################################################################
def predict_and_plot(hyperp, options, filepaths):

    #=== Mesh Properties ===#
    options.mesh_point_1 = [-1,-1]
    options.mesh_point_2 = [1,1]

    # options.nx = 15
    # options.ny = 15

    # options.nx = 30
    # options.ny = 30

    options.nx = 50
    options.ny = 50

    options.num_obs_points = 10
    options.order_fe_space = 1
    options.order_meta_space = 1
    options.num_nodes = (options.nx + 1) * (options.ny + 1)

    #=== Construct Mesh ===#
    fe_space, meta_space,\
    nodes, dof_fe, dof_meta = construct_mesh(options)

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

    epoch_list = np.arange(0,320,5)

    #for epoch in epoch_list:
    #   #=== Load Trained Neural Network ===#
    #   NN = VAE(hyperp, options,
    #           input_dimensions, latent_dimensions,
    #           None, None,
    #           positivity_constraint_log_exp)
    #   NN.load_weights(filepaths.directory_trained_NN + '_%d'%(epoch) + '/' +
    #           filepaths.NN_name)

    #   #=== Selecting Samples ===#
    #   sample_number = 128
    #   parameter_test_sample = np.expand_dims(parameter_test[sample_number,:], 0)
    #   state_obs_test_sample = np.expand_dims(state_obs_test[sample_number,:], 0)

    #   #=== Predictions ===#
    #   posterior_mean_pred, posterior_cov_pred = NN.encoder(state_obs_test_sample)
    #   posterior_pred_draw = NN.reparameterize(posterior_mean_pred, posterior_cov_pred)

    #   posterior_mean_pred = posterior_mean_pred.numpy().flatten()
    #   posterior_cov_pred = posterior_cov_pred.numpy().flatten()
    #   posterior_pred_draw = posterior_pred_draw.numpy().flatten()

    #   if options.model_aware == 1:
    #       state_obs_pred_draw = NN.decoder(np.expand_dims(posterior_pred_draw, 0))
    #       state_obs_pred_draw = state_obs_pred_draw.numpy().flatten()

    #   #=== Plotting Prediction ===#
    #   print('================================')
    #   print('      Plotting Predictions      ')
    #   print('================================')

    #   #=== Plot FEM Functions ===#
    #   cross_section_y = 0.0
    #   filename_extension = '_%d_%d.png'%(sample_number,epoch)
    #   plot_fem_function_fenics_2d(meta_space, posterior_mean_pred,
    #                               cross_section_y,
    #                               '',
    #                               filepaths.figure_posterior_mean + filename_extension,
    #                               (5,5), (0,6),
    #                               True)

    #   #=== Plot Cross-Section with Error Bounds ===#
    #   plot_cross_section(meta_space,
    #                   parameter_test_sample, posterior_mean_pred, posterior_cov_pred,
    #                   (-1,1), cross_section_y,
    #                   '',
    #                   filepaths.figure_parameter_cross_section + filename_extension,
    #                   (1.5,5.5))

    #   print('Predictions plotted')

    #=== Make Movie ===#
    sample_number = 128
    make_movie(filepaths.figure_posterior_mean + '_%d'%(sample_number),
               filepaths.directory_movie,
               'posterior_mean',
               2, 0, len(epoch_list))

    make_movie(filepaths.figure_parameter_cross_section + '_%d'%(sample_number),
               filepaths.directory_movie,
               'parameter_cross_section',
               2, 0, len(epoch_list))
