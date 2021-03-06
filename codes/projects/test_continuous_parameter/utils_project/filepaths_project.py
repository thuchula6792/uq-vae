'''Constructs filepaths for saving and loading project specific objects

This class contains the strings associated to project specific objects.
The majority of these strings are associated with loading the required objects.
However, this code also constructs the string case_name which identifies the
outputs of training, prediction and plotting.

You will need to specify:
    - directory_dataset: the directory containing your dataset
    - project_name: the name of your project
    - data_string: strings detailing the properties of the dataset
    - prior_string_: strings detailing the location of prior models used for
                     regularization, generating the training set and
                     generating the testing set
    - poi_: strings detailing the location of the training and testing set of
            the parameter-of-interest
    - qoi_: strings detailing the location of the training and testing set of
            the quantity-of-interest

Inputs:
    - options: dictionary storing the set options

Author: Hwan Goh, Oden Institute, Austin, Texas 2020
'''
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
        #=== Defining Filenames ===#
        data_options = 'm%d_n%d'%(options.mesh_dimensions, options.parameter_dimensions)
        directory_dataset = '../../../../../datasets/simple_1d/'
        if hasattr(options, 'continuous_linear_sin') and options.continuous_linear_sin == 1:
            project_name = 'continuous_linear_sin_1d'
        if hasattr(options, 'continuous_linear_uniform') and options.continuous_linear_uniform == 1:
            project_name = 'continuous_linear_uniform_1d'
        directory_dataset += project_name + '/' + data_options + '/'

        #=== Data Properties ===#
        if options.obs_type == 'full':
            obs_string = 'full'
        if options.obs_type == 'obs':
            obs_string = 'obs_o%d'%(options.num_obs_points)
        if options.add_noise == 1:
            noise_level_string = value_to_string(options.noise_level)
            noise_string = 'ns%s_%d'%(noise_level_string,options.num_obs_points)
        else:
            noise_string = 'ns0'
        data_string = data_options + '_' + obs_string + '_' + noise_string + '_'

        #=== Prior Properties ===#
        if options.prior_type_identity_reg == True:
            prior_string_reg = self.prior_string('identity',
                    options.prior_mean_identity_reg)
        if options.prior_type_identity_train == True:
            prior_string_train = self.prior_string('identity',
                    options.prior_mean_identity_train)
        if options.prior_type_identity_test == True:
            prior_string_test = self.prior_string('identity',
                    options.prior_mean_identity_test)

        if options.prior_type_laplacian_reg == True:
            prior_string_reg = self.prior_string_laplacian('laplacian',
                    options.prior_mean_laplacian_reg)
        if options.prior_type_laplacian_train == True:
            prior_string_train = self.prior_string_laplacian('laplacian',
                    options.prior_mean_laplacian_train)
        if options.prior_type_laplacian_test == True:
            prior_string_test = self.prior_string_laplacian('laplacian',
                    options.prior_mean_laplacian_test)

        #=== Case String ===#
        self.case_name = project_name + '_' + data_string + prior_string_train

        ################
        #   Datasets   #
        ################
        #=== Forward Operator ===#
        if hasattr(options, 'discrete_polynomial') and options.discrete_polynomial == 1:
            self.forward_operator = directory_dataset +\
                    project_name + '_forward_matrix_' +\
                    data_options
        if hasattr(options, 'continuous_linear_sin') and options.continuous_linear_sin == 1:
            self.forward_operator = directory_dataset +\
                    project_name + '_forward_vector_' +\
                    data_options
        if hasattr(options, 'continuous_linear_uniform') and options.continuous_linear_uniform == 1:
            self.forward_operator = directory_dataset +\
                    project_name + '_forward_vector_' +\
                    data_options

        #=== Parameters ===#
        self.obs_indices = directory_dataset +\
                project_name + '_' + 'obs_indices_' +\
                'o%d_'%(options.num_obs_points) + data_options
        self.poi_train = directory_dataset +\
                project_name + '_' + 'parameter_train_' +\
                'd%d_'%(options.num_data_train_load) + data_options + '_' + prior_string_train
        self.poi_test = directory_dataset +\
                project_name + '_' + 'parameter_test_' +\
                'd%d_'%(options.num_data_test_load) + data_options + '_' + prior_string_test
        self.poi_specific = ''

        #=== State ===#
        if options.obs_type == 'full':
            self.qoi_train = directory_dataset +\
                    project_name + '_' + 'state_' + options.obs_type + '_train_' +\
                    'd%d_'%(options.num_data_train_load) + data_options + '_' + prior_string_train
            self.qoi_test = directory_dataset +\
                    project_name + '_' + 'state_' + options.obs_type + '_test_' +\
                    'd%d_'%(options.num_data_test_load) + data_options + '_' + prior_string_test
            self.qoi_specific = ''
        if options.obs_type == 'obs':
            self.qoi_train = directory_dataset +\
                    project_name + '_' + 'state_' + options.obs_type + '_train_' +\
                    'o%d_d%d_' %(options.num_obs_points, options.num_data_train_load) +\
                    data_options + '_' + prior_string_train
            self.qoi_test = directory_dataset +\
                    project_name + '_' + 'state_' + options.obs_type + '_test_' +\
                    'o%d_d%d_' %(options.num_obs_points, options.num_data_test_load) +\
                    data_options + '_' + prior_string_test
            self.qoi_specific = ''

        #############
        #   Prior   #
        #############
        #=== Prior ===#
        self.prior_string_reg = prior_string_reg
        self.prior_mean = directory_dataset +\
                'prior_mean_' + data_options + '_' + prior_string_train
        self.prior_covariance = directory_dataset +\
                'prior_covariance_' + data_options + '_' + prior_string_train
        self.prior_covariance_inverse = directory_dataset +\
                'prior_covariance_inverse_' + data_options + '_' + prior_string_train
        self.prior_covariance_cholesky = directory_dataset +\
                'prior_covariance_cholesky_' + data_options + '_' + prior_string_train
        self.prior_covariance_cholesky_inverse = directory_dataset +\
                'prior_covariance_cholesky_inverse_' + data_options + '_' + prior_string_train

###############################################################################
#                               Prior Strings                                 #
###############################################################################
    def prior_string(self, prior_type, mean):
        mean_string = value_to_string(mean)
        return '%s_%s'%(prior_type, mean_string)

    def prior_string_laplacian(self, prior_type, mean):
        mean_string = value_to_string(mean)
        return '%s_%s'%(prior_type, mean_string)
