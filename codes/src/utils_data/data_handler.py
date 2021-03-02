import time

import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class DataHandler:
    def __init__(self,hyperp, options, filepaths,
                 obs_indices,
                 poi_dimensions, qoi_dimensions,
                 state_dimensions):

        #=== Filepaths ===#
        self.filepath_poi_train = filepaths.poi_train
        self.filepath_poi_test = filepaths.poi_test
        self.filepath_poi_specific = filepaths.poi_specific

        self.filepath_qoi_train = filepaths.qoi_train
        self.filepath_qoi_test = filepaths.qoi_test
        self.filepath_qoi_specific = filepaths.qoi_specific

        #=== Dimensions ===#
        self.poi_dimensions = poi_dimensions
        self.qoi_dimensions = qoi_dimensions
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
        self.poi_train, self.qoi_train = self.load_data(
                self.filepath_poi_train,
                self.filepath_qoi_train,
                self.num_data_train)
    def load_data_test(self):
        print('Loading Training Data')
        self.poi_test, self.qoi_test = self.load_data(
                self.filepath_poi_test,
                self.filepath_qoi_test,
                self.num_data_test)

    def load_data_specific(self):
        print('Loading Training Data')
        self.poi_specific, self.qoi_specific = self.load_data(
                self.filepath_poi_specific,
                self.filepath_qoi_specific,
                1)

    def load_data(self, filepath_poi_data, filepath_qoi_data,
                  num_data):

        start_time_load_data = time.time()

        df_poi_data = pd.read_csv(filepath_poi_data + '.csv')
        poi_data = df_poi_data.to_numpy()
        poi_data = poi_data.reshape((-1,self.poi_dimensions))
        poi_data = poi_data[0:num_data,:]
        poi_data = poi_data.astype(np.float32)

        df_qoi_data = pd.read_csv(filepath_qoi_data + '.csv')
        qoi_data = df_qoi_data.to_numpy()
        qoi_data = qoi_data.reshape((-1,self.qoi_dimensions))
        qoi_data = qoi_data[0:num_data,:]
        qoi_data = qoi_data.astype(np.float32)

        elapsed_time_load_data = time.time() - start_time_load_data
        print('Time taken to load data: %.4f' %(elapsed_time_load_data))

        return poi_data, qoi_data

###############################################################################
#                                 Add Noise                                   #
###############################################################################
    def add_noise_qoi_train(self):
        self.qoi_train_max = np.max(self.qoi_train)
        self.qoi_train = self.add_noise(self.qoi_train, self.qoi_train_max)
    def add_noise_qoi_test(self):
        self.qoi_test_max = np.max(self.qoi_test)
        self.qoi_test = self.add_noise(self.qoi_test, self.qoi_test_max)
    def add_noise_qoi_specific(self):
        self.qoi_specific_max = np.max(self.qoi_specific)
        self.qoi_specific = self.add_noise(self.qoi_specific, self.qoi_specific_max)

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
        return self.construct_noise_regularization_matrix(self.qoi_train, self.qoi_train_max)
    def construct_noise_regularization_matrix_test(self):
        return self.construct_noise_regularization_matrix(self.qoi_test, self.qoi_test_max)

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
    def normalize_data_poi_train(self):
        self.poi_train = self.normalize_data(self.poi_train)
    def normalize_data_qoi_train(self):
        self.qoi_train = self.normalize_data(self.qoi_train)
    def normalize_data_poi_test(self):
        self.poi_test = self.normalize_data(self.poi_test)
    def normalize_data_qoi_test(self):
        self.qoi_test = self.normalize_data(self.qoi_test)

    def normalize_data(self, data):
        data = data/np.expand_dims(np.linalg.norm(data, ord = 2, axis = 1), 1)

        return data
