'''Manipulates prior related objects

This class contains methods associated with manipulation of the training and
testing dataset. Currently, these methods involve loading the prior mean,
covariance, inverse of the covariance, Cholesky of the covariance and inverse of
the Cholesky of the covariance.

Inputs:
    - hyperp: dictionary storing set hyperparameter values
    - options: dictionary storing the set options
    - filepaths: class instance storing the filepaths
    - poi_dimensions: dimension of the parameter-of-interest

Author: Hwan Goh, Oden Institute, Austin, Texas 2020
'''
import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class PriorHandler:
    def __init__(self, hyperp, options, filepaths, poi_dimensions):

        self.filepath_prior_mean = filepaths.prior_mean
        self.filepath_prior_covariance = filepaths.prior_covariance
        self.filepath_prior_covariance_inverse = filepaths.prior_covariance_inverse
        self.filepath_prior_covariance_cholesky = filepaths.prior_covariance_cholesky
        self.filepath_prior_covariance_cholesky_inverse = filepaths.prior_covariance_cholesky_inverse

        self.poi_dimensions = poi_dimensions

    def load_prior_mean(self):
        return self.load_vector(self.filepath_prior_mean)

    def load_prior_covariance(self):
        return self.load_matrix(self.filepath_prior_covariance)

    def load_prior_covariance_inverse(self):
        return self.load_matrix(self.filepath_prior_covariance_inverse)

    def load_prior_covariance_cholesky(self):
        return self.load_matrix(self.filepath_prior_covariance_cholesky)

    def load_prior_covariance_cholesky_inverse(self):
        return self.load_matrix(self.filepath_prior_covariance_cholesky_inverse)

    def load_vector(self, filepath):
        df_vector = pd.read_csv(filepath + '.csv')
        vector = df_vector.to_numpy()
        return vector.astype(np.float32).flatten()

    def load_matrix(self, filepath):
        df_matrix = pd.read_csv(filepath + '.csv')
        matrix = df_matrix.to_numpy()
        matrix = matrix.reshape((self.poi_dimensions, self.poi_dimensions))
        return matrix.astype(np.float32)
