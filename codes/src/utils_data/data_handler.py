import time

import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class DataHandler:
    def __init__(self,hyperp, options, filepaths,
                 obs_indices,
                 input_dimensions, output_dimensions,
                 state_dimensions):

        #=== Filepaths ===#
        self.filepath_input_train = filepaths.input_train
        self.filepath_input_test = filepaths.input_test
        self.filepath_input_specific = filepaths.input_specific

        self.filepath_output_train = filepaths.output_train
        self.filepath_output_test = filepaths.output_test
        self.filepath_output_specific = filepaths.output_specific

        #=== Dimensions ===#
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.state_dimensions = state_dimensions

        #=== Dataset Sizes ===#
        self.num_data_train = hyperp.num_data_train
        self.num_data_test = options.num_data_test

        #=== Noise Options ===#
        self.noise_level = options.noise_level
        self.obs_indices = obs_indices
        self.num_obs_points = options.num_obs_points
        self.dampening_scalar = 0.001

        #=== Random Seed ===#
        self.random_seed = options.random_seed

###############################################################################
#                                 Load Data                                   #
###############################################################################
    def load_data_train(self):
        print('Loading Training Data')
        self.input_train, self.output_train = self.load_data(
                self.filepath_input_train,
                self.filepath_output_train,
                self.num_data_train)
    def load_data_test(self):
        print('Loading Training Data')
        self.input_test, self.output_test = self.load_data(
                self.filepath_input_test,
                self.filepath_output_test,
                self.num_data_test)

    def load_data_specific(self):
        print('Loading Training Data')
        self.input_specific, self.output_specific = self.load_data(
                self.filepath_input_specific,
                self.filepath_output_specific,
                1)

    def load_data(self, filepath_input_data, filepath_output_data,
                  num_data):

        start_time_load_data = time.time()

        df_input_data = pd.read_csv(filepath_input_data + '.csv')
        input_data = df_input_data.to_numpy()
        input_data = input_data.reshape((-1,self.input_dimensions))
        input_data = input_data[0:num_data,:]
        input_data = input_data.astype(np.float32)

        df_output_data = pd.read_csv(filepath_output_data + '.csv')
        output_data = df_output_data.to_numpy()
        output_data = output_data.reshape((-1,self.output_dimensions))
        output_data = output_data[0:num_data,:]
        output_data = output_data.astype(np.float32)

        elapsed_time_load_data = time.time() - start_time_load_data
        print('Time taken to load data: %.4f' %(elapsed_time_load_data))

        return input_data, output_data

###############################################################################
#                                 Add Noise                                   #
###############################################################################
    def add_noise_output_train(self):
        self.output_train_max = np.max(self.output_train)
        self.output_train = self.add_noise(self.output_train, self.output_train_max)
    def add_noise_output_test(self):
        self.output_test_max = np.max(self.output_test)
        self.output_test = self.add_noise(self.output_test, self.output_test_max)
    def add_noise_output_specific(self):
        self.output_specific_max = np.max(self.output_specific)
        self.output_specific = self.add_noise(self.output_specific, self.output_specific_max)

    def add_noise(self, data, data_max):
        #=== Add Noise ===#
        np.random.seed(self.random_seed)
        noise = np.random.normal(0, 1, data.shape)
        data += self.noise_level*data_max*noise

        return data

###############################################################################
#                         Noise Regularization Matrix                         #
###############################################################################
    def construct_noise_regularization_matrix_train(self):
        return self.construct_noise_regularization_matrix(self.output_train, self.output_train_max)
    def construct_noise_regularization_matrix_test(self):
        return self.construct_noise_regularization_matrix(self.output_test, self.output_test_max)

    def construct_noise_regularization_matrix(self, data, data_max):
        #=== Noise Regularization Matrix ===#
        np.random.seed(self.random_seed)
        diagonal = 1/(self.noise_level*data_max)*np.ones(data.shape[1])

        return diagonal.astype(np.float32)

###############################################################################
#                              Measurement Matrix                             #
###############################################################################
    def construct_measurement_matrix(self):
        measurement_matrix = np.zeros(
                                (self.num_obs_points, self.state_dimensions),
                                dtype = np.float32)
        for obs_ind in range(0, self.num_obs_points):
            measurement_matrix[obs_ind, self.obs_indices[obs_ind]] = 1
        return measurement_matrix

###############################################################################
#                               Normalize Data                                #
###############################################################################
    def normalize_data_input_train(self):
        self.input_train = self.normalize_data(self.input_train)
    def normalize_data_output_train(self):
        self.output_train = self.normalize_data(self.output_train)
    def normalize_data_input_test(self):
        self.input_test = self.normalize_data(self.input_test)
    def normalize_data_output_test(self):
        self.output_test = self.normalize_data(self.output_test)

    def normalize_data(self, data):
        data = data/np.expand_dims(np.linalg.norm(data, ord = 2, axis = 1), 1)

        return data
