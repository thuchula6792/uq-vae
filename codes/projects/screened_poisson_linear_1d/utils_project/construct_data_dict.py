'''Constructs project specific dictionary containing dataset related objects

To construct the dictionary, the code will create an instance of the DataHandler
class. Utilizing the methods of this class then generates the objects to be
stored in the dictionary. Suggested methods of this class to utilize are:
    1) loading the training and testing dataset
    2) adding noise to the training/testing dataset
    3) form the noise regularization matrix
    4) construct the measurement matrix for the case where
       the parameter-to-observable map is linear

Inputs:
    - hyperp: dictionary storing set hyperparameter values
    - options: dictionary storing the set options
    - filepaths: class instance storing the filepaths

Author: Hwan Goh, Oden Institute, Austin, Texas 2020
'''
import numpy as np
import pandas as pd

from utils_data.data_handler import DataHandler

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_data_dict(hyperp, options, filepaths):

    #=== Load Observation Indices ===#
    if options.obs_type == 'full':
        obs_dimensions = options.parameter_dimensions
        obs_indices = []
    if options.obs_type == 'obs':
        obs_dimensions = options.num_obs_points
        print('Loading Boundary Indices')
        df_obs_indices = pd.read_csv(filepaths.project.obs_indices + '.csv')
        obs_indices = df_obs_indices.to_numpy()

    #=== Prepare Data ===#
    data = DataHandler(hyperp, options, filepaths,
                       obs_indices,
                       options.parameter_dimensions, obs_dimensions,
                       options.parameter_dimensions)
    data.load_data_train()
    data.load_data_test()
    if options.add_noise == True:
        data.add_noise_qoi_train()
        data.add_noise_qoi_test()
        noise_regularization_matrix = data.construct_noise_regularization_matrix_train()
        noise_regularization_matrix = np.expand_dims(noise_regularization_matrix, axis=0)
    else:
        noise_regularization_matrix = np.ones((1,obs_dimensions), dtype=np.float32)
    measurement_matrix = data.construct_measurement_matrix()

    #=== Construct Dictionary ===#
    data_dict = {}
    data_dict["obs_dimensions"] = obs_dimensions
    data_dict["obs_indices"] = obs_indices
    data_dict["parameter_train"] = data.poi_train
    data_dict["state_obs_train"] = data.qoi_train
    data_dict["parameter_test"] = data.poi_test
    data_dict["state_obs_test"] = data.qoi_test
    data_dict["noise_regularization_matrix"] = noise_regularization_matrix
    data_dict["measurement_matrix"] = measurement_matrix

    return data_dict
