#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  28 10:16:28 2020
@author: hwan
"""

from utils_io.value_to_string import value_to_string

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                              Project File Paths                             #
###############################################################################
class FilePathsProject:
    def __init__(self, options):

        ################
        #   Case Name  #
        ################
        #=== Key Strings ===#
        data_options = 'n%d'%(options.parameter_dimensions)
        if options.boundary_conditions_dirichlet == True:
            project_name = 'screened_poisson_linear_dirichlet_1d_'
            directory_dataset =\
                    '../../../../../datasets/fenics/screened_poisson_linear_dirichlet_1d/' +\
                    data_options + '/'
        if options.boundary_conditions_neumann == True:
            project_name = 'screened_poisson_linear_neumann_1d_'
            directory_dataset =\
                    '../../../../../datasets/fenics/screened_poisson_linear_neumann_1d/' +\
                    data_options + '/'

        #=== Data Type ===#
        if options.obs_type == 'full':
            obs_string = 'full'
        if options.obs_type == 'obs':
            obs_string = 'obs_o%d'%(options.num_obs_points)
        if options.add_noise == 1:
            noise_level_string = value_to_string(options.noise_level)
            noise_string = 'ns%s_%d'%(noise_level_string,options.num_noisy_obs)
        else:
            noise_string = 'ns0'
        data_string = data_options + '_' + obs_string + '_' + noise_string + '_'

        #=== Prior Properties ===#
        if options.prior_type_lp_reg == True:
            prior_string_reg = self.prior_string_lp('lp',
                    options.prior_mean_lp_reg,
                    options.prior_gamma_lp_reg,
                    options.prior_delta_lp_reg)
        if options.prior_type_lp_train == True:
            prior_string_train = self.prior_string_lp('lp',
                    options.prior_mean_lp_train,
                    options.prior_gamma_lp_train,
                    options.prior_delta_lp_train)
        if options.prior_type_lp_test == True:
            prior_string_test = self.prior_string_lp('lp',
                    options.prior_mean_lp_test,
                    options.prior_gamma_lp_test,
                    options.prior_delta_lp_test)

        #=== Case String ===#
        self.case_name = project_name + data_string + prior_string_train

        ################
        #   Datasets   #
        ################
        #=== Parameters ===#
        self.obs_indices = directory_dataset +\
                project_name + 'obs_indices_' +\
                'o%d_'%(options.num_obs_points) + data_options
        self.input_train = directory_dataset +\
                project_name +\
                'parameter_train_' +\
                'd%d_'%(options.num_data_train_load) + data_options + '_' + prior_string_train
        self.input_test = directory_dataset +\
                project_name +\
                'parameter_test_' +\
                'd%d_'%(options.num_data_test_load) + data_options + '_' + prior_string_test
        self.input_specific = directory_dataset +\
                project_name + 'parameter_specific_' + data_options
        if options.obs_type == 'full':
            self.output_train = directory_dataset +\
                    project_name +\
                    'state_' + options.obs_type + '_train_' +\
                    'd%d_'%(options.num_data_train_load) + data_options + '_' + prior_string_train
            self.output_test = directory_dataset +\
                    project_name +\
                    'state_' + options.obs_type + '_test_' +\
                    'd%d_'%(options.num_data_test_load) + data_options + '_' + prior_string_test
            self.output_specific = directory_dataset +\
                                project_name + 'state_' + options.obs_type + '_specific_' +\
                                data_options
        if options.obs_type == 'obs':
            self.output_train = directory_dataset +\
                    project_name +\
                    'state_' + options.obs_type + '_train_' +\
                    'o%d_d%d_' %(options.num_obs_points, options.num_data_train_load) +\
                    data_options + '_' + prior_string_train
            self.output_test = directory_dataset +\
                    project_name +\
                    'state_' + options.obs_type + '_test_' +\
                    'o%d_d%d_' %(options.num_obs_points, options.num_data_test_load) +\
                    data_options + '_' + prior_string_test
            self.output_specific = directory_dataset +\
                                project_name + 'state_' + options.obs_type + '_specific_' +\
                                'o%d_'%(options.num_obs_points) +\
                                data_options

        #############
        #   Prior   #
        #############
        #=== Prior ===#
        if options.prior_type_identity_reg == True:
            self.prior_string_reg = 'identity'
        else:
            self.prior_string_reg = prior_string_reg
        self.prior_mean = directory_dataset +\
                'prior_mean_' + data_options + '_' + prior_string_reg
        self.prior_covariance = directory_dataset +\
                'prior_covariance_' + data_options + '_' + prior_string_reg
        self.prior_covariance_cholesky = directory_dataset +\
                'prior_covariance_cholesky_' + data_options + '_' + prior_string_reg
        self.prior_covariance_cholesky_inverse = directory_dataset +\
                'prior_covariance_cholesky_inverse_' + data_options + '_' + prior_string_reg

        ###################
        #   FEM Objects   #
        ###################
        #=== Forward Operator ===#
        self.forward_matrix = directory_dataset +\
                'forward_matrix_' + data_options
        self.mass_matrix = directory_dataset +\
                'mass_matrix_' + data_options

###############################################################################
#                               Prior Strings                                 #
###############################################################################
    def prior_string_lp(self, prior_type, mean, gamma, delta):
        mean_string = value_to_string(mean)
        gamma_string = value_to_string(gamma)
        delta_string = value_to_string(delta)

        return '%s_%s_%s_%s'%(prior_type, mean_string, gamma_string, delta_string)
