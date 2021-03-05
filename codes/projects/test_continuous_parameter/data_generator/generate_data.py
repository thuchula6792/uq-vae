#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Generates training and testing data

The parameter-of-interest is a discretization of a continuous function on a one
dimensional domain.

Whilst you can toggle what objects are generated,
all generated objects will be loaded by default.

In preparation for generating data, the code will:
    1) Instantiate an Options class containing the set options
    2) Construct the project specific FilePaths class from the options
       attributes

You will need to specify:
    - In Options:
        - Whether to construct and save the forward operator matrices
        - Whether to construct the prior related objects
        - Whether to draw and save parameters from the prior
        - The forward operator
        - Whether to generate training or testing data
        - Whether to plot the drawn parameters and corresponding states

Outputs will be stored in the directories designated by the FilePaths class

Author: Hwan Goh, Oden Institute, Austin, Texas 2020
'''
import os
import sys
sys.path.insert(0, os.path.realpath('../../../src'))
sys.path.insert(0, os.path.realpath('..'))

import numpy as np
import pandas as pd

# Import src code
from utils_io.value_to_string import value_to_string

# Import data generator codes
from filepaths import FilePaths
from prior_laplace_finite_difference import prior_laplace_finite_difference
from prior_io import save_prior, load_prior
from forward_functions import continuous_linear
from dataset_io import save_parameter, save_state, load_parameter
from plot_1d import plot_1d

# Import project utilities
from utils_project.construct_system_matrices_continuous_linear import\
        construct_system_matrices, load_system_matrices

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                   Options                                   #
###############################################################################
class Options:
    #=== IO Options ===%
    construct_and_save_matrices = True
    construct_prior = True
    draw_and_save_parameters = True

    #=== Data Type ===#
    continuous_linear_sin = False
    continuous_linear_uniform = True

    #=== Select Train or Test Set ===#
    generate_train_data = True
    generate_test_data = False

    #=== Data Properties ===#
    num_data = 5000
    mesh_dimensions = 50
    parameter_dimensions = mesh_dimensions
    num_obs_points = 1

    #=== Identity Prior ===#
    prior_type_identity = True
    prior_mean_identity = 1

    #=== Laplacian Prior ===#
    prior_type_laplacian = False
    prior_mean_laplacian = 1

    #=== Plotting ===#
    plot_parameters = False
    plot_states = False
    plot_parameters_y_axis_min = 0
    plot_parameters_y_axis_max = 2
    plot_states_y_axis_min = 0
    plot_states_y_axis_max = 10

    #=== Random Seed ===#
    random_seed = 4

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":
    ##################
    #   Setting Up   #
    ##################
    #=== Run Options ===#
    options = Options()

    #=== File Paths ===#
    filepaths = FilePaths(options)

    ###############################
    #   Generate and Save Prior   #
    ###############################
    #=== Mesh ===#
    mesh = np.linspace(0, 1, options.mesh_dimensions, endpoint = True)

    #=== Construct and Save Prior ===#
    if options.construct_prior == True and options.generate_test_data == False:
        if options.prior_type_identity == True:
            prior_mean = options.prior_mean_identity*np.ones(options.parameter_dimensions)
            prior_covariance = np.eye(options.parameter_dimensions)
            prior_covariance_inverse = np.eye(options.parameter_dimensions)
            prior_covariance_cholesky = np.eye(options.parameter_dimensions)
            prior_covariance_cholesky_inverse = np.eye(options.parameter_dimensions)
        if options.prior_type_laplacian == True:
            prior_mean = options.prior_mean_laplacian*np.ones(options.parameter_dimensions)
            prior_covariance,\
            prior_covariance_cholesky,\
            prior_covariance_cholesky_inverse\
            = prior_laplace_finite_difference(mesh)
        save_prior(filepaths, prior_mean, prior_covariance, prior_covariance_inverse,
                   prior_covariance_cholesky, prior_covariance_cholesky_inverse)

    ######################
    #   Forward Matrix   #
    ######################
    if options.construct_and_save_matrices == 1:
        if not os.path.exists(filepaths.directory_dataset):
            os.makedirs(filepaths.directory_dataset)
        construct_system_matrices(filepaths, options, mesh)
    forward_vector = load_system_matrices(options, filepaths)

    ##############################
    #   Generate and Save Data   #
    ##############################
    #=== Draw From Prior ===#
    if options.draw_and_save_parameters == 1:
        parameter = np.zeros((options.num_data, options.parameter_dimensions))
        prior_mean, _, _, prior_covariance_cholesky, _ = load_prior(filepaths,
                                                                 options.parameter_dimensions)
        for m in range(0, options.num_data):
            epsilon = np.random.normal(0, 1, options.parameter_dimensions)
            parameter[m,:] = np.matmul(prior_covariance_cholesky, epsilon) + prior_mean.T
        if not os.path.exists(filepaths.directory_dataset):
            os.makedirs(filepaths.directory_dataset)
        df_parameter= pd.DataFrame({'parameter': parameter.flatten()})
        df_parameter.to_csv(filepaths.parameter + '.csv', index=False)

    #=== Load Parameters ===#
    parameter = load_parameter(filepaths, options.parameter_dimensions, options.num_data)

    #=== Plot Parameters ===#
    if options.plot_parameters == True:
        for m in range(0, options.num_data):
            plot_1d(parameter[m,:], filepaths.directory_figures + '/' + 'parameters_%d.png'%(m),
                    options.parameter_dimensions,
                    options.plot_parameters_y_axis_min, options.plot_parameters_y_axis_max,
                    '', 'Parameter Number', 'Value')

    #=== Generate State ===#
    state = np.zeros((options.num_data, 1))
    for m in range(0, options.num_data):
        state[m] = continuous_linear(parameter[m,:], forward_vector)

    #=== Generate Observation Data ===#
    np.random.seed(options.random_seed)
    obs_indices = np.sort(
            np.random.choice(
                range(0, 1), options.num_obs_points, replace = False))
    state_obs = state[:,obs_indices]

    #=== Plot States ===#
    if options.plot_states == True:
        plot_1d(state, filepaths.directory_figures + '/' + 'states.png',
                options.num_data,
                options.plot_states_y_axis_min, options.plot_states_y_axis_max,
                '', 'Datapoint', 'Value')

    #=== Save Dataset ====#
    save_parameter(filepaths, parameter)
    save_state(filepaths, obs_indices, state, state_obs)
    print('Data Generated and Saved')
