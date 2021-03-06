#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Calls the training routine for Bayesian optimization of hyperparameters

The parameter-to-observable map is modelled is required to be linear
The package scikit-opt is used for Bayesian hyperparameter optimization

In preparation for optimization of hyperparameters, the code will:
    1) Construct a dictionary containing the set hyperparameter values
       read from the .yaml file
    2) Construct a dictionary containing the set options
       read from the .yaml file
    3) Construct the project specific as well as neural-network related
       FilePaths classes from the hyperp and options dictionaries
    4) Construct a dictionary containing loaded training and testing data
       and related objects
    5) Construct a dictionary containing loaded prior related objects

You will need to specify:
    - In add_options():
        - Whether to use distributed training
        - Which gpus to utilize
        - Whether to delete the trained suboptimal neural networks
    - Number of neural networks of to be trained through setting the number of
      calls of the training routine
    - The hyperparameters of interest and their corresponding ranges.
      Unspecified hyperparameters will obtain a default value from that set in the
      hyperp_.yaml file

Outputs will be stored in the following directories:
    - uq-vae/hyperparameter_optimization/ which contains:
        - /outputs/ for metrics and convergence data
        - /trained_nns/ for the optimal trained network and associated training metrics
        - /tensorboard/ for Tensorboard training metrics of the optimal network

Author: Hwan Goh, Oden Institute, Austin, Texas 2021
'''
import os
import shutil

import numpy as np
import pandas as pd

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

# Import src code
from utils_hyperparameter_optimization.hyperparameter_optimization_output\
        import output_results

# Import skopt code
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt import dump, load

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                     Hyperparameter Optimization Routine                     #
###############################################################################
def optimize_hyperparameters(hyperp, options, filepaths,
                             n_calls, space, hyperp_of_interest_dict,
                             data_dict, prior_dict,
                             training, loss_val_index,
                             FilePathsClass, project_paths):

    ############################
    #   Objective Functional   #
    ############################
    @use_named_args(space)
    def objective_functional(**hyperp_of_interest_dict):
        #=== Assign Hyperparameters of Interest ===#
        for key, value in hyperp_of_interest_dict.items():
            hyperp[key] = value

        #=== Update File Paths with New Hyperparameters ===#
        filepaths = FilePathsClass(hyperp, options, project_paths)

        #=== Training Routine ===#
        training(hyperp, options, filepaths,
                 data_dict, prior_dict)

        #=== Loading Metrics For Output ===#
        print('Loading Metrics')
        df_metrics = pd.read_csv(filepaths.trained_nn + "_metrics" + '.csv')
        array_metrics = df_metrics.to_numpy()
        storage_array_loss_val = array_metrics[:,loss_val_index]

        return storage_array_loss_val[-1]

    ################################
    #   Optimize Hyperparameters   #
    ################################
    hyperp_opt_result = gp_minimize(objective_functional, space,
                                    n_calls=n_calls, random_state=None)

    ######################
    #   Output Results   #
    ######################
    output_results(filepaths, hyperp_of_interest_dict, hyperp_opt_result)

    #####################################################
    #   Delete All Suboptimal Trained Neural Networks   #
    #####################################################
    if options.delete_suboptimal_nn == True:
        #=== Assigning hyperp with Optimal Hyperparameters ===#
        for num, key in enumerate(hyperp_of_interest_dict.keys()):
            hyperp[key] = hyperp_opt_result.x[num]

        #=== Updating File Paths with Optimal Hyperparameters ===#
        filepaths = FilePathsClass(hyperp, options, project_paths)

        #=== Deleting Suboptimal Neural Networks ===#
        directories_list_trained_nn = os.listdir(
                path=filepaths.directory_hyperp_opt_trained_nn_case)

        for filename in directories_list_trained_nn:
            if filename != filepaths.nn_name:
                shutil.rmtree(filepaths.directory_hyperp_opt_trained_nn_case + '/' + filename)
                shutil.rmtree(filepaths.directory_hyperp_opt_tensorboard_case + '/' + filename)

        print('Suboptimal Trained Networks Deleted')
