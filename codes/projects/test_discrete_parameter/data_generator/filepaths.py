
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  28 10:16:28 2020

@author: hwan
"""
import os

from utils_io.value_to_string import value_to_string

###############################################################################
#                              Project File Paths                             #
###############################################################################
class FilePaths:
    def __init__(self, options):

        ################
        #   Case Name  #
        ################
        #=== Defining Filenames ===#
        data_options = 'm%d_n%d'%(options.mesh_dimensions, options.parameter_dimensions)
        self.directory_dataset = '../../../../../datasets/simple_1d/'
        if hasattr(options, 'discrete_polynomial') and options.discrete_polynomial == 1:
            project_name = 'discrete_polynomial_1d'
        if hasattr(options, 'discrete_exponential') and options.discrete_exponential == 1:
            project_name = 'discrete_exponential_1d'
        self.directory_dataset += project_name + '/' + data_options + '/'
        if not os.path.exists(self.directory_dataset):
                os.makedirs(self.directory_dataset)

        #=== Train of Test ===#
        if options.generate_train_data == 1:
            train_or_test = 'train_'
        if options.generate_test_data == 1:
            train_or_test = 'test_'

        #=== Prior Properties ===#
        if hasattr(options, 'prior_type_identity') and options.prior_type_identity == 1:
            prior_string = self.prior_string('identity',
                    options.prior_mean_identity)
        if hasattr(options, 'prior_type_diag') and options.prior_type_diag == 1:
            prior_string = self.prior_string_diag('diag',
                    options.prior_mean_diag)
        if hasattr(options, 'prior_type_full') and options.prior_type_full == 1:
            prior_string = self.prior_string_full('full',
                    options.prior_mean_full)
        if hasattr(options, 'prior_type_laplacian') and options.prior_type_laplacian == 1:
            prior_string = self.prior_string('laplacian',
                    options.prior_mean_laplacian)

        ################
        #   Datasets   #
        ################
        #=== Forward Operator ===#
        self.forward_matrix = self.directory_dataset +\
                project_name + '_forward_matrix_' +\
                data_options

        #=== Parameters ===#
        self.parameter = self.directory_dataset +\
                project_name + '_parameter_' + train_or_test +\
                'd%d_' %(options.num_data) + data_options + '_' + prior_string

        #=== State ===#
        self.obs_indices = self.directory_dataset +\
                project_name + '_obs_indices_' +\
                'o%d_'%(options.num_obs_points) + data_options
        self.state_full = self.directory_dataset +\
                project_name + '_state_full_' + train_or_test +\
                'd%d_' %(options.num_data) +\
                data_options + '_' + prior_string
        self.state_obs = self.directory_dataset +\
                project_name + '_state_obs_' + train_or_test +\
                'o%d_d%d_'%(options.num_obs_points, options.num_data) +\
                data_options + '_' + prior_string

        #=== Prior ===#
        self.prior_mean = self.directory_dataset +\
                'prior_mean_' + data_options + '_' + prior_string
        self.prior_covariance = self.directory_dataset +\
                'prior_covariance_' + data_options + '_' + prior_string
        self.prior_covariance_inverse = self.directory_dataset +\
                'prior_covariance_inverse_' + data_options + '_' + prior_string
        self.prior_covariance_cholesky = self.directory_dataset +\
                'prior_covariance_cholesky_' + data_options + '_' + prior_string
        self.prior_covariance_cholesky_inverse = self.directory_dataset +\
                'prior_covariance_cholesky_inverse_' + data_options + '_' + prior_string

        #=== Figures ==#
        self.directory_figures = 'Figures/'
        if not os.path.exists(self.directory_figures):
            os.makedirs(self.directory_figures)

###############################################################################
#                               Prior Strings                                 #
###############################################################################
    def prior_string_diag(self, prior_type, mean):
        mean_string = value_to_string(mean)

        return '%s_%s'%(prior_type, mean_string)

    def prior_string_full(self, prior_type, mean):
        mean_string = value_to_string(mean)

        return '%s_%s'%(prior_type, mean_string)
