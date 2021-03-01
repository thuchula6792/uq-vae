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
        project_name = 'poisson_3d_'
        data_options = 'n%d' %(options.parameter_dimensions)
        directory_dataset = '../../../../../datasets/fenics/poisson_3d/' +\
                'n%d/'%(options.parameter_dimensions)

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
        if options.prior_type_AC_train == True:
            prior_string_train = self.prior_string_AC('AC',
                    options.prior_mean_AC_train,
                    options.prior_variance_AC_train,
                    options.prior_corr_AC_train)
        if options.prior_type_AC_test == True:
            prior_string_test = self.prior_string_AC('AC',
                    options.prior_mean_AC_test,
                    options.prior_variance_AC_test,
                    options.prior_corr_AC_test)

        if options.prior_type_matern_train == True:
            prior_string_train = prior_string_matern('matern',
                    options.prior_kern_type_train,
                    options.prior_cov_length_train)
        if options.prior_type_matern_test == True:
            prior_string_test = prior_string_matern('matern',
                    options.prior_kern_type_test,
                    options.prior_cov_length_test)

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
                project_name + 'parameter_blob_' + data_options
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
        self.prior_mean = directory_dataset +\
                'prior_mean_' + data_options + '_' + prior_string_train
        self.prior_covariance = directory_dataset +\
                'prior_covariance_' + data_options + '_' + prior_string_train
        self.prior_covariance_cholesky = directory_dataset +\
                'prior_covariance_cholesky_' + data_options + '_' + prior_string_train
        self.prior_covariance_cholesky_inverse = directory_dataset +\
                'prior_covariance_cholesky_inverse_' + data_options + '_' + prior_string_train

        ###################
        #   FEM Objects   #
        ###################
        #=== Pre-Matrices ===#
        self.premass = directory_dataset +\
                'premass_' + data_options
        self.prestiffness = directory_dataset +\
                'prestiffness_' + data_options
        self.boundary_matrix = directory_dataset +\
                'boundary_matrix_' + data_options
        self.load_vector = directory_dataset +\
                'load_vector_' + data_options

        #=== Mesh ===# For plotting FEM function
        mesh_name = 'mesh_square_2D_n%d' %(options.parameter_dimensions)
        mesh_directory = '../../../../../Datasets/Mesh/' + mesh_name + '/'
        self.mesh_nodes = mesh_directory + mesh_name + '_nodes.csv'
        self.mesh_elements = mesh_directory + mesh_name + '_elements.csv'
        self.mesh_boundary_indices_edges = mesh_directory + mesh_name +\
                '_boundary_indices_edges.csv'
        self.mesh_boundary_indices = mesh_directory + mesh_name +\
                '_boundary_indices.csv'
        self.mesh_boundary_indices_bottom = mesh_directory + mesh_name +\
                '_boundary_indices_bottom.csv'
        self.mesh_boundary_indices_left = mesh_directory + mesh_name +\
                '_boundary_indices_left.csv'
        self.mesh_boundary_indices_right = mesh_directory + mesh_name +\
                '_boundary_indices_right.csv'
        self.mesh_boundary_indices_top = mesh_directory + mesh_name +\
                '_boundary_indices_top.csv'

###############################################################################
#                               Prior Strings                                 #
###############################################################################
    def prior_string_AC(self, prior_type, mean, variance, corr):
        mean_string = value_to_string(mean)
        variance_string = value_to_string(variance)
        corr_string = value_to_string(corr)

        return '%s_%s_%s_%s'%(prior_type, mean_string, variance_string, corr_string)

    def prior_string_matern(self, prior_type, kern_type, cov_length):
        cov_length_string = value_to_string(cov_length)

        return '%s_%s_%s'%(prior_type, kern_type, cov_length)
