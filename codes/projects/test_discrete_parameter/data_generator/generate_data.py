#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Generates training and testing data

The parameter-of-interest discrete; often coefficients of the forward function

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
from prior_io import save_prior, load_prior
from forward_functions import discrete_polynomial, discrete_exponential
from dataset_io import save_parameter, save_state, load_parameter
from plot_1d import plot_1d

# Import project utilities
from utils_project.construct_system_matrices_discrete_polynomial import\
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
    discrete_polynomial = True
    discrete_exponential = False

    #=== Select Train or Test Set ===#
    generate_train_data = False
    generate_test_data = True

    #=== Data Properties ===#
    num_data = 200
    mesh_dimensions = 100
    parameter_dimensions = 2
    num_obs_points = 20

    #=== Full Prior ===#
    prior_type_diag = False
    prior_mean_diag = 0

    #=== Diagonal Prior ===#
    prior_type_full = True
    prior_mean_full = 4

    #=== Plotting ===#
    plot_parameters = False
    plot_states = False
    plot_parameters_y_axis_min = -1
    plot_parameters_y_axis_max = 8
    plot_states_y_axis_min = -0
    plot_states_y_axis_max = 50

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
        if options.prior_type_diag == 1:
            prior_mean = options.prior_mean_diag*np.ones(options.parameter_dimensions)
            prior_covariance = np.diag(
                                np.random.normal(0,1,size=(options.parameter_dimensions)))
        if options.prior_type_full == 1:
            prior_mean = options.prior_mean_full*np.ones(options.parameter_dimensions)
            rand_matrix = np.random.normal(0,1,
                            size=(options.parameter_dimensions, options.parameter_dimensions))
            prior_covariance = np.finfo(np.float32).eps*np.identity(options.parameter_dimensions) +\
                            np.matmul(rand_matrix, np.transpose(rand_matrix))
        prior_covariance_inverse = np.linalg.inv(prior_covariance)
        prior_covariance_cholesky = np.linalg.cholesky(prior_covariance)
        prior_covariance_cholesky_inverse = np.linalg.inv(prior_covariance_cholesky)
        save_prior(filepaths, prior_mean,
                   prior_covariance, prior_covariance_inverse,
                   prior_covariance_cholesky, prior_covariance_cholesky_inverse)

    ######################
    #   Forward Matrix   #
    ######################
    if options.discrete_polynomial == 1:
        if options.construct_and_save_matrices == 1:
            if not os.path.exists(filepaths.directory_dataset):
                os.makedirs(filepaths.directory_dataset)
            construct_system_matrices(filepaths, options, mesh)
        forward_matrix = load_system_matrices(options, filepaths)

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
    state = np.zeros((options.num_data, options.mesh_dimensions))
    for m in range(0, options.num_data):
        if options.discrete_polynomial == 1:
            state[m,:], _ = discrete_polynomial(parameter[m,:], forward_matrix,
                                                options.parameter_dimensions)
        if options.discrete_exponential == 1:
            state[m,:], _ = discrete_exponential(parameter[m,:], mesh,
                                                options.parameter_dimensions)

    #=== Generate Observation Data ===#
    np.random.seed(options.random_seed)
    obs_indices = np.sort(
            np.random.choice(
                range(0, options.mesh_dimensions), options.num_obs_points, replace = False))
    state_obs = state[:,obs_indices]

    #=== Plot States ===#
    if options.plot_states == True:
        for m in range(0, options.num_data):
            plot_1d(state[m,:], filepaths.directory_figures + '/' + 'state_%d.png'%(m),
                    options.mesh_dimensions,
                    options.plot_states_y_axis_min, options.plot_states_y_axis_max,
                    '', 'Mesh Coordinate', 'Value')

    #=== Save Dataset ====#
    save_parameter(filepaths, parameter)
    save_state(filepaths, obs_indices, state, state_obs)
    print('Data Generated and Saved')
