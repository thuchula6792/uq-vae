#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:47:02 2019

@author: hwan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off

###############################################################################
#                                Plot Metrics                                 #
###############################################################################
def plot_and_save_metrics(hyperp, options, filepaths):
    print('================================')
    print('        Plotting Metrics        ')
    print('================================')

    dpi=100

    #=== Load Metrics ===#
    print('Loading Metrics')
    df_metrics = pd.read_csv(filepaths.trained_NN + "_metrics" + '.csv')
    array_metrics = df_metrics.to_numpy()

    ####################
    #   Load Metrics   #
    ####################
    storage_array_loss_train = array_metrics[:,0]
    storage_array_loss_train_VAE = array_metrics[:,1]
    storage_array_loss_train_encoder = array_metrics[:,2]
    storage_array_relative_error_input_VAE = array_metrics[:,10]
    storage_array_relative_error_latent_encoder = array_metrics[:,11]
    storage_array_relative_error_input_decoder = array_metrics[:,12]
    # storage_array_relative_gradient_norm = array_metrics[:,13]

    ################
    #   Plotting   #
    ################
    #=== Loss Train ===#
    fig_loss = plt.figure()
    x_axis = np.linspace(1, hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    plt.plot(x_axis, np.log(storage_array_loss_train))
    plt.title('Log-Loss for Training Neural Network')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    figures_savefile_name = filepaths.directory_figures + '/' +\
            'loss.png'
    plt.savefig(figures_savefile_name, dpi=dpi)
    plt.close(fig_loss)

    #=== Loss Autoencoder ===#
    fig_loss = plt.figure()
    x_axis = np.linspace(1, hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    plt.plot(x_axis, np.log(storage_array_loss_train_VAE))
    plt.title('Log-Loss for VAE')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    figures_savefile_name = filepaths.directory_figures + '/' +\
            'loss_autoencoder.png'
    plt.savefig(figures_savefile_name, dpi=dpi)
    plt.close(fig_loss)

    #=== Loss Encoder ===#
    fig_loss = plt.figure()
    x_axis = np.linspace(1, hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    plt.plot(x_axis, np.log(storage_array_loss_train_encoder))
    plt.title('Log-Loss for Encoder')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    figures_savefile_name = filepaths.directory_figures + '/' +\
            'loss_encoder.png'
    plt.savefig(figures_savefile_name, dpi=dpi)
    plt.close(fig_loss)

    #=== Relative Error Autoencoder ===#
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1,hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    plt.plot(x_axis, storage_array_relative_error_input_VAE)
    plt.title('Relative Error for Autoencoder')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    figures_savefile_name = filepaths.directory_figures + '/' +\
            'relative_error_autoencoder.png'
    plt.savefig(figures_savefile_name, dpi=dpi)
    plt.close(fig_accuracy)

    #=== Relative Error Encoder ===#
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1,hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    plt.plot(x_axis, storage_array_relative_error_latent_encoder)
    plt.title('Relative Error for Encoder')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    figures_savefile_name = filepaths.directory_figures + '/' +\
            'relative_error_encoder.png'
    plt.savefig(figures_savefile_name, dpi=dpi)
    plt.close(fig_accuracy)

    #=== Relative Error Decoder ===#
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1,hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    plt.plot(x_axis, storage_array_relative_error_input_decoder)
    plt.title('Relative Error for Decoder')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    figures_savefile_name = filepaths.directory_figures + '/' +\
            'relative_error_decoder.png'
    plt.savefig(figures_savefile_name, dpi=dpi)
    plt.close(fig_accuracy)

    #=== Relative Gradient Norm ===#
    # fig_gradient_norm = plt.figure()
    # x_axis = np.linspace(1,hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    # plt.plot(x_axis, storage_array_relative_gradient_norm)
    # plt.title('Relative Gradient Norm')
    # plt.xlabel('Epochs')
    # plt.ylabel('Relative Error')
    # figures_savefile_name = filepaths.directory_figures + '/' +\
    #         'relative_error_gradient_norm.png'
    # plt.savefig(figures_savefile_name)
    # plt.close(fig_gradient_norm)

    #if options.model_augmented == 1:
    #    #=== Relative Error Decoder ===#
    #    fig_loss = plt.figure()
    #    x_axis = np.linspace(1,hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    #    plt.plot(x_axis, storage_array_loss_train_forward_model)
    #    plt.title('Log-loss Forward Model')
    #    plt.xlabel('Epochs')
    #    plt.ylabel('Relative Error')
    #    figures_savefile_name = filepaths.directory_figures + '/' +\
    #            'loss_forward_model.png'
    #    plt.savefig(figures_savefile_name)
    #    plt.close(fig_loss)

    print('Plotting complete')
