#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  28 10:16:28 2020

@author: hwan
"""

from utils_io.value_to_string import value_to_string

###############################################################################
#                              Project File Paths                             #
###############################################################################
class FilePathsProject:
    def __init__(self, options):

        ################
        #   Case Name  #
        ################
        #=== Key Strings ===#
        project_name = 'advection_diffusion_2d_'
        if options.flow_navier_stokes == True:
            flow_string = 'navier_stokes'
        if options.flow_darcy == True:
            flow_string = 'darcy'
        if options.time_stepping_erk4 == True:
            time_stepping_string = 'erk4'
        if options.time_stepping_lserk4 == True:
            time_stepping_string = 'lserk4'
        if options.time_stepping_implicit == True:
            time_stepping_string = 'imp'
        num_nodes_string = 'n%d'%(options.parameter_dimensions)
        data_options = num_nodes_string + '_' +\
                       flow_string + '_' +\
                       time_stepping_string
        directory_dataset = '../../../../../datasets/fenics/advection_diffusion_2d/' +\
            num_nodes_string + '/' + flow_string + '_' + time_stepping_string + '/'

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
        if options.prior_type_blp_reg == True:
            prior_string_reg = self.prior_string_blp('blp',
                                    options.prior_mean_blp_reg,
                                    options.prior_gamma_blp_reg,
                                    options.prior_delta_blp_reg)
        if options.prior_type_blp_train == True:
            prior_string_train = self.prior_string_blp('blp',
                                    options.prior_mean_blp_train,
                                    options.prior_gamma_blp_train,
                                    options.prior_delta_blp_train)
        if options.prior_type_blp_test == True:
            prior_string_test = self.prior_string_blp('blp',
                                    options.prior_mean_blp_test,
                                    options.prior_gamma_blp_test,
                                    options.prior_delta_blp_test)

        if options.prior_type_ac_reg == True:
            prior_string_reg = self.prior_string_ac('ac',
                                    options.prior_mean_ac_reg,
                                    options.prior_variance_ac_reg,
                                    options.prior_corr_ac_reg)
        if options.prior_type_ac_train == True:
            prior_string_train = self.prior_string_ac('ac',
                                    options.prior_mean_ac_train,
                                    options.prior_variance_ac_train,
                                    options.prior_corr_ac_train)
        if options.prior_type_ac_test == True:
            prior_string_test = self.prior_string_ac('ac',
                                    options.prior_mean_ac_test,
                                    options.prior_variance_ac_test,
                                    options.prior_corr_ac_test)

        #=== Case String ===#
        self.case_name = project_name + data_string + prior_string_train

        ################
        #   Datasets   #
        ################
        #=== Parameters ===#
        self.obs_indices = directory_dataset +\
                project_name + 'obs_indices_' +\
                'o%d_'%(options.num_obs_points) + num_nodes_string
        self.input_train = directory_dataset +\
                project_name +\
                'parameter_train_' +\
                'd%d_'%(options.num_data_train_load) + num_nodes_string + '_' + prior_string_train
        self.input_test = directory_dataset +\
                project_name +\
                'parameter_test_' +\
                'd%d_'%(options.num_data_test_load) + num_nodes_string + '_' + prior_string_test
        self.input_specific = directory_dataset +\
                project_name + 'parameter_blob_' + num_nodes_string
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
                                project_name + 'state_' + options.obs_type + '_blob_' +\
                                'o%d_'%(options.num_obs_points) +\
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
                                project_name + 'state_' + options.obs_type + '_blob_' +\
                                'o%d_'%(options.num_obs_points) +\
                                data_options

        #############
        #   Prior   #
        #############
        #=== Prior ===#
        self.prior_string_reg = prior_string_reg
        self.prior_mean = directory_dataset +\
                'prior_mean_' + num_nodes_string + '_' + prior_string_reg
        self.prior_covariance = directory_dataset +\
                'prior_covariance_' + num_nodes_string + '_' + prior_string_reg
        self.prior_covariance_cholesky = directory_dataset +\
                'prior_covariance_cholesky_' + num_nodes_string + '_' + prior_string_reg
        self.prior_covariance_cholesky_inverse = directory_dataset +\
                'prior_covariance_cholesky_inverse_' + num_nodes_string + '_' + prior_string_reg

        ###################
        #   FEM Objects   #
        ###################
        #=== FEM Operators ===#
        self.fem_operator_spatial = directory_dataset +\
                'fem_operator_spatial_' + num_nodes_string
        self.fem_operator_implicit_ts = directory_dataset +\
                'fem_operator_implicit_ts_' + num_nodes_string
        self.fem_operator_implicit_ts_rhs = directory_dataset +\
                'fem_operator_implicit_ts_rhs_' + num_nodes_string

###############################################################################
#                               Prior Strings                                 #
###############################################################################
    def prior_string_blp(self, prior_type, mean, gamma, delta):
        mean_string = value_to_string(mean)
        gamma_string = value_to_string(gamma)
        delta_string = value_to_string(delta)

        return '%s_%s_%s_%s'%(prior_type, mean_string, gamma_string, delta_string)

    def prior_string_ac(self, prior_type, mean, variance, corr):
        mean_string = value_to_string(mean)
        variance_string = value_to_string(variance)
        corr_string = value_to_string(corr)

        return '%s_%s_%s_%s'%(prior_type, mean_string, variance_string, corr_string)
