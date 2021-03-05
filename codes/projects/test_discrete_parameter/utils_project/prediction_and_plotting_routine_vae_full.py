#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:17:53 2019
@author: hwan
"""
import sys

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off
import scipy.stats as st

# Import src code
from utils_data.data_handler import DataHandler
from utils_data.prior_handler import PriorHandler
from neural_networks.nn_vae_full import VAE
from utils_misc.positivity_constraints import positivity_constraint_log_exp

# Import project utilities
from utils_project.get_forward_operators_tf import load_forward_operator_tf
from utils_project.solve_forward_1d import SolveForward1D
from utils_project.plot_bivariate_gaussian import plot_bivariate_gaussian

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                              Plot Predictions                               #
###############################################################################
def predict_and_plot(hyperp, options, filepaths):

    #=== Load Observation Indices ===#
    if options.obs_type == 'full':
        obs_dimensions = options.mesh_dimensions
        obs_indices = []
    if options.obs_type == 'obs':
        obs_dimensions = options.num_obs_points
        print('Loading Boundary Indices')
        df_obs_indices = pd.read_csv(filepaths.project.obs_indices + '.csv')
        obs_indices = df_obs_indices.to_numpy()

    #=== Data and Latent Dimensions of Autoencoder ===#
    input_dimensions = obs_dimensions
    latent_dimensions = options.parameter_dimensions

    #=== Prepare Data ===#
    data = DataHandler(hyperp, options, filepaths,
                        obs_indices,
                        options.parameter_dimensions, obs_dimensions,
                        options.mesh_dimensions)
    data.load_data_test()
    if options.add_noise == 1:
        data.add_noise_qoi_test()
    parameter_test = data.poi_test
    state_obs_test = data.qoi_test

    #=== Load Trained Neural Network ===#
    NN = VAE(hyperp, options,
                input_dimensions, latent_dimensions,
                None, None,
                tf.identity)
    NN.load_weights(filepaths.trained_NN)

    #=== Construct Forward Model ===#
    forward_operator = load_forward_operator_tf(options, filepaths)
    forward_model =\
            SolveForward1D(options, filepaths, forward_operator, obs_indices)
    if options.discrete_polynomial == True:
        forward_model_solve = forward_model.discrete_polynomial
    if options.discrete_exponential == True:
        forward_model_solve = forward_model.discrete_exponential

    #=== Selecting Samples ===#
    sample_number = 4
    parameter_test_sample = np.expand_dims(parameter_test[sample_number,:], 0)
    state_obs_test_sample = np.expand_dims(state_obs_test[sample_number,:], 0)

    #=== Predictions ===#
    post_mean_pred, log_post_std_pred, post_cov_chol_pred = NN.encoder(state_obs_test_sample)
    n_samples = 1000
    posterior_pred_draws = np.zeros((n_samples, post_mean_pred.shape[1]),
                                dtype=np.float32)
    state_obs_pred_draws = np.zeros((n_samples, state_obs_test_sample.shape[1]),
                                dtype=np.float32)
    for n in range(0,n_samples):
        posterior_pred_draws[n,:] = NN.reparameterize(post_mean_pred, post_cov_chol_pred)
    if options.model_aware == True:
        state_obs_pred_draws = NN.decoder(posterior_pred_draws)
    else:
        state_obs_pred_draws = forward_model_solve(posterior_pred_draws)

    #=== Plotting Prediction ===#
    print('================================')
    print('      Plotting Predictions      ')
    print('================================')
    n_bins = 100
    for n in range(0, post_mean_pred.shape[1]):
        #=== Posterior Histogram ===#
        plt.hist(posterior_pred_draws[:,n], density=True,
                    range=[3,5], bins=n_bins)
        #=== True Parameter Value ===#
        plt.axvline(parameter_test_sample[0,n], color='r',
                linestyle='dashed', linewidth=3,
                label="True Parameter Value")
        #=== Predicted Posterior Mean ===#
        plt.axvline(post_mean_pred[0,n], color='b',
                linestyle='dashed', linewidth=1,
                label="Predicted Posterior Mean")
        #=== Probability Density Function ===#
        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 301)
        kde = st.gaussian_kde(posterior_pred_draws[:,n])
        #=== Title and Labels ===#
        plt.plot(kde_xs, kde.pdf(kde_xs))
        plt.legend(loc="upper left")
        plt.ylabel('Probability')
        plt.xlabel('Parameter Value')
        plt.title("Marginal Posterior Parameter_%d"%(n));
        #=== Save and Close Figure ===#
        plt.savefig(filepaths.figure_parameter_pred + '_%d'%(n))
        plt.close()

    print('Predictions plotted')

###############################################################################
#                              Compare Covariance                             #
###############################################################################
    #=== Construct Likelihood Matrix ===#
    if options.add_noise == True:
        noise_regularization_matrix = data.construct_noise_regularization_matrix_test()
        noise_regularization_matrix = np.expand_dims(noise_regularization_matrix, axis=0)
    else:
        noise_regularization_matrix = np.ones((1,obs_dimensions), dtype=np.float32)
    measurement_matrix = data.construct_measurement_matrix()
    likelihood_matrix = tf.linalg.matmul(
                            tf.transpose(tf.linalg.matmul(measurement_matrix,forward_operator)),
                            tf.linalg.matmul(
                                tf.linalg.diag(tf.squeeze(noise_regularization_matrix)),
                                tf.linalg.matmul(measurement_matrix,forward_operator)))

    #=== Construct Inverse of Prior Matrix ===#
    prior = PriorHandler(hyperp, options, filepaths,
                        options.parameter_dimensions)
    prior_mean = prior.load_prior_mean()
    prior_cov_inv = prior.load_prior_covariance_inverse()

    #=== Construct True Posterior ===#
    post_cov_true = np.linalg.inv(likelihood_matrix + prior_cov_inv)
    post_cov_pred = np.matmul(
                        np.reshape(post_cov_chol_pred,
                            (options.parameter_dimensions,options.parameter_dimensions)),
                        np.transpose(np.reshape(post_cov_chol_pred,
                            (options.parameter_dimensions,options.parameter_dimensions))))

    #=== Construct True Mean ===#
    post_mean_true = np.matmul(post_cov_true,
                        np.matmul(
                            np.transpose(np.matmul(measurement_matrix,forward_operator)),
                            np.matmul(np.diag(noise_regularization_matrix.flatten()),
                                      np.transpose(state_obs_test_sample))) +\
                        np.matmul(prior_cov_inv, np.expand_dims(prior_mean, axis=1)))

    #=== Relative Error of Matrices ===#
    relative_error = np.linalg.norm(post_cov_true - post_cov_pred, ord='fro')\
                    /np.linalg.norm(post_cov_true, ord='fro')
    print('relative error = %.4f'%(relative_error))

    #=== Bivariate Contour Plot ===#
    if options.parameter_dimensions == 2:
        plot_bivariate_gaussian(filepaths.figure_bivariate_pred,
                                post_mean_pred.numpy(), post_cov_pred,
                                (5,5), 3.8, 5.2, (0,14),
                                '', 'u_1', 'u_2')
        plot_bivariate_gaussian(filepaths.figure_bivariate_true,
                                post_mean_true, post_cov_true,
                                (5,5), 3.8, 5.2, (0,14),
                                '', 'u_1', 'u_2')
