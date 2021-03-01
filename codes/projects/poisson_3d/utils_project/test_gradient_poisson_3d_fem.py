import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from positivity_constraints import positivity_constraint_log_exp
from utils_misc.get_FEM_matrices_tf import load_FEM_matrices_tf
from utils_project.FEM_prematrices_poisson_2D import FEMPrematricesPoisson2D
from neural_networks.nn_ae_fwd_inv import AutoencoderFwdInv
from loss_and_relative_errors import loss_penalized_difference

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def test_gradient(hyperp, options, filepaths):
###############################################################################
#                        Construct Directional Derivative                     #
###############################################################################
    if options.reverse_autoencoder == 1:
        print("Gradient test not coded for reverse autoencoder")
        return 0

    ##################
    #   Load Prior   #
    ##################
    prior_mean, _, prior_covariance_cholesky, _\
    = load_prior(options, filepaths,
                 load_mean = 1,
                 load_covariance = 0,
                 load_covariance_cholesky = 1, load_covariance_cholesky_inverse = 0)
    k = 0.5

    #####################
    #   Generate Data   #
    #####################
    #=== Draw True Parameter ===#
    normal_draw = np.random.normal(0, 1, options.parameter_dimensions)
    parameter_true = np.matmul(prior_covariance_cholesky, normal_draw) + prior_mean.T
    parameter_true = positivity_constraint_log_exp(parameter_true)
    parameter_true = tf.cast(parameter_true, tf.float32)
    parameter_true = tf.expand_dims(parameter_true, axis = 0)

    #=== Load Observation Indices ===#
    if options.obs_type == 'full':
        obs_dimensions = options.parameter_dimensions
        obs_indices = []
    if options.obs_type == 'obs':
        obs_dimensions = options.num_obs_points
        print('Loading Boundary Indices')
        df_obs_indices = pd.read_csv(filepaths.obs_indices_savefilepath + '.csv')
        obs_indices = df_obs_indices.to_numpy()

    #=== Load FEM Matrices ===#
    _, prestiffness, boundary_matrix, load_vector =\
            load_fem_matrices_tf(options, filepaths,
                                 load_premass = 0,
                                 load_prestiffness = 1)

    #=== Construct Forward Model ===#
    forward_model = FEMPrematricesPoisson2D(options, filepaths,
                                            obs_indices,
                                            prestiffness,
                                            boundary_matrix, load_vector)

    #=== Generate Observation Data ===#
    state_obs_true = forward_model.solve_PDE_prematrices_sparse(parameter_true)

    ########################
    #   Compute Gradient   #
    ########################
    #=== Data and Latent Dimensions of Autoencoder ===#
    if options.standard_autoencoder == 1:
        input_dimensions = options.parameter_dimensions
        latent_dimensions = obs_dimensions
    if options.reverse_autoencoder == 1:
        input_dimensions = obs_dimensions
        latent_dimensions = options.parameter_dimensions

    #=== Form Neural Network ===#
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    bias_initializer = 'zeros'
    NN = AutoencoderFwdInv(hyperp, options,
                           input_dimensions, latent_dimensions,
                           kernel_initializer, bias_initializer,
                           positivity_constraint_log_exp)

    #=== Display Neural Network Architecture ===#
    NN.build((hyperp.batch_size, input_dimensions))
    NN.summary()

    #=== Draw and Set Weights ===#
    weights_list = []
    for n in range(0, len(NN.weights)):
        weights_list.append(tf.random.normal(NN.weights[n].shape, 0, 0.05, tf.float32))
    NN.set_weights(weights_list)

    #=== Compute Gradient ===#
    with tf.GradientTape() as tape:
        NN_output = NN(parameter_true)
        test = positivity_constraint_log_exp(NN_output)
        forward_model_pred = forward_model.solve_PDE_prematrices_sparse(
                positivity_constraint_log_exp(NN_output))
        loss_0 = loss_penalized_difference(state_obs_true, forward_model_pred, 1)
    gradients = tape.gradient(loss_0, NN.trainable_variables)

    ############################
    #   Direction Derivative   #
    ############################
    #=== Draw Direction ===#
    directions_list = []
    for n in range(0, len(gradients)):
        directions_list.append(tf.random.normal(gradients[n].shape, 0, 0.05, tf.float32))
        directions_list[n] = directions_list[n]/tf.linalg.norm(directions_list[n],2)

    #=== Directional Derivative ===#
    directional_derivative = 0.0
    for n in range(0, len(gradients)):
        directional_derivative += tf.reduce_sum(tf.multiply(gradients[n], directions_list[n]))
    directional_derivative = directional_derivative.numpy()

###############################################################################
#                     Construct Finite Difference Derivative                  #
###############################################################################
    loss_h_list = []
    gradients_fd_list = []
    errors_list = []
    h_collection = np.power(2., -np.arange(32))

    for h in h_collection:
        #=== Perturbed Loss ===#
        weights_perturbed_list = []
        for n in range(0, len(NN.weights)):
            weights_perturbed_list.append(weights_list[n] + h*directions_list[n])
        NN.set_weights(weights_perturbed_list)
        NN_perturbed_output = NN(parameter_true)
        forward_model_perturbed_pred = forward_model.solve_PDE_prematrices_sparse(
                positivity_constraint_log_exp(NN_perturbed_output))

        loss_h = loss_penalized_difference(state_obs_true, forward_model_perturbed_pred, 1)
        gradient_fd = (loss_h - loss_0)/h
        error = abs(gradient_fd - directional_derivative)/abs(directional_derivative)

        loss_h = loss_h.numpy()
        gradient_fd = gradient_fd.numpy()
        error = error.numpy()

        loss_h_list.append(loss_h)
        gradients_fd_list.append(gradient_fd)
        errors_list.append(error)

###############################################################################
#                                   Plotting                                  #
###############################################################################
    #=== Plot Functional ===#
    plt.loglog(h_collection, loss_h_list, "-ob", label="Functional")
    plt.savefig('functional.png', dpi=200)
    plt.close()

    #=== Plot Error ===#
    plt.loglog(h_collection, errors_list, "-ob", label="Error")
    plt.loglog(h_collection,
            (.5*errors_list[0]/h_collection[0])*h_collection, "-.k", label="First Order")
    plt.savefig('grad_test.png', dpi=200)
    plt.cla()
    plt.clf()

    print(f"FD gradients: {gradients_fd_list}")
    print(f"Errors: {errors_list}")
