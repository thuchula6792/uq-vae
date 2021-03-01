import numpy as np
import pandas as pd

from utils_data.prior_handler import PriorHandler

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_prior_dict(hyperp, options, filepaths,
                         load_mean = True,
                         load_covariance = True,
                         load_covariance_cholesky = True,
                         load_covariance_cholesky_inverse = True):

    prior_dict = {}
    prior = PriorHandler(hyperp, options, filepaths,
                         options.parameter_dimensions)

    #=== Prior Mean ===#
    if load_mean == True:
        prior_mean = prior.load_prior_mean()
        prior_dict["prior_mean"] = np.expand_dims(prior_mean,0)

    #=== Prior Covariance ===#
    if load_covariance == True:
        prior_covariance = prior.load_prior_covariance()
        prior_dict["prior_covariance"] = prior_covariance

    #=== Prior Covariance Cholesky ===#
    if load_covariance_cholesky == True:
        prior_covariance_cholesky = prior.load_prior_covariance_cholesky()
        prior_dict["prior_covariance_cholesky"] = prior_covariance_cholesky

    #=== Prior Covariance Cholesky Inverse ===#
    if load_covariance_cholesky_inverse == True:
        prior_covariance_cholesky_inverse = prior.load_prior_covariance_cholesky_inverse()
        prior_dict["prior_covariance_cholesky_inverse"] = prior_covariance_cholesky_inverse

    #=== Identity Prior ===#
    if options.prior_type_identity_reg == True:
        prior_dict["prior_covariance"] = np.identity(
                options.parameter_dimensions, dtype = np.float32)
        prior_dict["prior_covariance_cholesky"] = np.identity(
                options.parameter_dimensions, dtype = np.float32)
        prior_dict["prior_covariance_cholesky_inverse"] = np.identity(
                options.parameter_dimensions, dtype = np.float32)

    return prior_dict
