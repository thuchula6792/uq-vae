#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:16:28 2019

@author: hwan
"""
import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class PriorHandler:
    def __init__(self, hyperp, options, filepaths, input_dimensions):

        self.filepath_prior_mean = filepaths.prior_mean
        self.filepath_prior_covariance = filepaths.prior_covariance
        self.filepath_prior_covariance_cholesky = filepaths.prior_covariance_cholesky
        self.filepath_prior_covariance_cholesky_inverse = filepaths.prior_covariance_cholesky_inverse

        self.input_dimensions = input_dimensions

    def load_prior_mean(self):
        return self.load_vector(self.filepath_prior_mean)

    def load_prior_covariance(self):
        return self.load_matrix(self.filepath_prior_covariance)

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
        matrix = matrix.reshape((self.input_dimensions, self.input_dimensions))
        return matrix.astype(np.float32)
