#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Collection of terms that form loss functionals

Author: Hwan Goh, Oden Institute, Austin, Texas 2020
'''
import numpy as np
import tensorflow as tf
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                               General Loss                                  #
###############################################################################
def loss_penalized_difference(true, pred, penalty):
    '''penalized squared error of the true and predicted values'''
    return penalty*true.shape[1]*tf.keras.losses.mean_squared_error(true, pred)

def loss_weighted_penalized_difference(true, pred, weight_matrix, penalty):
    '''weighted penalized squared error of the true and predicted values'''
    if len(pred.shape) == 1:
        pred = tf.expand_dims(pred, axis=1)
    if weight_matrix.shape[0] == weight_matrix.shape[1]: # If matrix is square
        return penalty*true.shape[1]*tf.keras.losses.mean_squared_error(
                tf.linalg.matmul(true, tf.transpose(weight_matrix)),
                tf.linalg.matmul(pred, tf.transpose(weight_matrix)))
    else: # Diagonal weight matrices with diagonals stored as rows
        return penalty*true.shape[1]*tf.keras.losses.mean_squared_error(
                tf.multiply(weight_matrix, true),
                tf.multiply(weight_matrix, pred))

###############################################################################
#                      Loss Diagonal Posterior Covariance                     #
###############################################################################
def loss_kld(post_mean, log_post_var,
             prior_mean, prior_cov_inv,
             penalty):
    '''Kullback-Leibler divergence between the model posterior and the prior
    model for the case where the model posterior possesses a diagonal covariance
    matrix
    '''
    trace_prior_cov_inv_times_cov_post = tf.reduce_sum(
            tf.multiply(tf.linalg.diag_part(prior_cov_inv), tf.math.exp(log_post_var)),
            axis=1)
    prior_weighted_prior_mean_minus_post_mean = tf.reduce_sum(
            tf.multiply(tf.transpose(prior_mean - post_mean),
                tf.linalg.matmul(prior_cov_inv, tf.transpose(prior_mean - post_mean))),
            axis = 0)
    return penalty*(trace_prior_cov_inv_times_cov_post
            + prior_weighted_prior_mean_minus_post_mean
            - tf.math.reduce_sum(log_post_var, axis=1))

###############################################################################
#                        Loss Full Posterior Covariance                       #
###############################################################################
def loss_weighted_post_cov_full_penalized_difference(true, pred,
                                                     post_cov_chol,
                                                     penalty):
    '''Monte-Carlo estimate of the Kullback-Leibler divergence
    between the true posterior and the model posterior for the
    case where the model posterior possesses a full covariance
    matrix
    '''
    batched_value = weighted_inner_product_chol_solve(
                        tf.transpose(tf.reshape(
                            post_cov_chol[0,:], (true.shape[1], true.shape[1]))),
                        tf.expand_dims(true[0,:] - pred[0,:], axis=1))
    for m in range(1, true.shape[0]):
        batched_value = tf.concat(
                [batched_value,
                weighted_inner_product_chol_solve(
                    tf.transpose(tf.reshape(
                        post_cov_chol[m,:], (true.shape[1], true.shape[1]))),
                    tf.expand_dims(true[m,:] - pred[m,:], axis=1))], axis=0)
    return penalty*batched_value

def weighted_inner_product_chol_solve(weight_matrix, vector):
    '''Evaluates data-misfit term weighted by the inverse of the full model
    posterior covariance
    '''
    return tf.linalg.matmul(tf.transpose(vector),
                tf.linalg.solve(tf.transpose(weight_matrix),
                                tf.linalg.solve(weight_matrix, vector)))

def loss_trace_likelihood(post_cov_chol,
                          identity_otimes_likelihood_matrix,
                          penalty):
    '''For the case where the parameter-to-observable map is linear, the
    expectation of the likelihood does not require a Monte-Carlo approximation
    and so there is an extra trace term which is computed by this function
    '''
    return penalty*tf.linalg.matmul(
                post_cov_chol,
                identity_otimes_likelihood_matrix.matmul(tf.transpose(post_cov_chol)))

def loss_kld_full(post_mean, log_post_std, post_cov_chol,
                  prior_mean, prior_cov_inv, identity_otimes_prior_cov_inv,
                  penalty):
    '''Kullback-Leibler divergence between the model posterior and the prior
    model for the case where the model posterior possesses a full covariance
    matrix
    '''
    trace_prior_cov_inv_times_cov_post = tf.linalg.matmul(
                post_cov_chol,
                identity_otimes_prior_cov_inv.matmul(tf.transpose(post_cov_chol)))
    prior_weighted_prior_mean_minus_post_mean = tf.reduce_sum(
            tf.multiply(tf.transpose(prior_mean - post_mean),
                tf.linalg.matmul(prior_cov_inv, tf.transpose(prior_mean - post_mean))),
            axis = 0)
    return penalty*(trace_prior_cov_inv_times_cov_post
            + prior_weighted_prior_mean_minus_post_mean
            - 2*tf.math.reduce_sum(log_post_std, axis=1))

###############################################################################
#                             Loss Forward Model                              #
###############################################################################
def loss_forward_model(hyperp, options,
                       forward_model,
                       state_obs_true, parameter_pred,
                       penalty):
    '''Computes the expectation of the likelihood using the modelled
    parameter-to-observable map
    '''
    forward_model_state_pred = forward_model(parameter_pred)
    forward_model_state_pred = tf.cast(forward_model_state_pred, dtype=tf.float32)
    return penalty*state_obs_true.shape[1]*tf.keras.losses.mean_squared_error(state_obs_true,
            forward_model_state_pred)

###############################################################################
#                               Relative Error                                #
###############################################################################
def relative_error(true, pred):
    '''relative error between testing data and prediction'''
    return tf.keras.losses.mean_absolute_percentage_error(true, pred)
