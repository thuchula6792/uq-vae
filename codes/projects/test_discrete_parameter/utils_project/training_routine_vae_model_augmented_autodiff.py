import os
import sys

import tensorflow as tf
import numpy as np
import pandas as pd

# Import src code
from utils_training.form_train_val_test import form_train_val_test_tf_batches
from neural_networks.nn_vae import VAE
from optimize.optimize_vae_model_augmented_autodiff import optimize
from optimize.optimize_distributed_vae_model_augmented_autodiff import optimize_distributed
from utils_misc.positivity_constraints import positivity_constraint_exp,\
                                              positivity_constraint_log_exp

# Import project utilities
from utils_project.get_forward_operators_tf import load_forward_operator_tf
from utils_project.solve_forward_1d import SolveForward1D

import pdb

###############################################################################
#                                  Training                                   #
###############################################################################
def training(hyperp, options, filepaths,
             data_dict, prior_dict):

    #=== GPU Settings ===#
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if options.distributed_training == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = options.which_gpu
    if options.distributed_training == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = options.dist_which_gpus
        gpus = tf.config.experimental.list_physical_devices('GPU')

    #=== Construct Validation Set and Batches ===#
    input_and_latent_train, input_and_latent_val, input_and_latent_test,\
    num_batches_train, num_batches_val, num_batches_test\
    = form_train_val_test_tf_batches(
            data_dict["state_obs_train"], data_dict["parameter_train"],
            data_dict["state_obs_test"], data_dict["parameter_test"],
            hyperp.batch_size, options.random_seed)

    #=== Data and Latent Dimensions of Autoencoder ===#
    input_dimensions = data_dict["obs_dimensions"]
    latent_dimensions = options.parameter_dimensions

    #=== Load Forward Operator  ===#
    forward_operator  = load_forward_operator_tf(options, filepaths)

    #=== Construct Forward Model ===#
    forward_model = SolveForward1D(options, filepaths, forward_operator, data_dict["obs_indices"])
    if options.discrete_polynomial == True:
        forward_model_solve = forward_model.discrete_polynomial
    if options.discrete_exponential == True:
        forward_model_solve = forward_model.discrete_exponential

    #=== Neural Network Regularizers ===#
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    bias_initializer = 'zeros'

    #=== Non-distributed Training ===#
    if options.distributed_training == 0:
        #=== Neural Network ===#
        nn = VAE(hyperp, options,
                 input_dimensions, latent_dimensions,
                 kernel_initializer, bias_initializer,
                 positivity_constraint_log_exp)

        #=== Optimizer ===#
        optimizer = tf.keras.optimizers.Adam()

        #=== Training ===#
        optimize(hyperp, options, filepaths,
                 nn, optimizer,
                 input_and_latent_train, input_and_latent_val, input_and_latent_test,
                 input_dimensions, latent_dimensions, num_batches_train,
                 data_dict["noise_regularization_matrix"],
                 prior_dict["prior_mean"], prior_dict["prior_covariance_inverse"],
                 forward_model_solve)

    #=== Distributed Training ===#
    if options.distributed_training == 1:
        dist_strategy = tf.distribute.MirroredStrategy()
        with dist_strategy.scope():
            #=== Neural Network ===#
            nn = VAE(hyperp, options,
                     input_dimensions, latent_dimensions,
                     kernel_initializer, bias_initializer,
                     positivity_constraint_log_exp)

            #=== Optimizer ===#
            optimizer = tf.keras.optimizers.Adam()

        #=== Training ===#
        optimize_distributed(dist_strategy,
                hyperp, options, filepaths,
                nn, optimizer,
                input_and_latent_train, input_and_latent_val, input_and_latent_test,
                input_dimensions, latent_dimensions, num_batches_train,
                data_dict["noise_regularization_matrix"],
                prior_dict["prior_mean"], prior_dict["prior_covariance_inverse"],
                forward_model_solve)
