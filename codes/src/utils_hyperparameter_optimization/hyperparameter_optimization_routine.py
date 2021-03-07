#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Bayesian hyperparameter optimization routine

This script contains the function objective_functional() that is the target loss
functional for Bayesian hyperparameter optimization.

The function objective_functional() will:
    1) Update a dictionary with the new hyperparameter values
    2) Update the neural-network related FilePaths classes using the
       new values in the updated hyperp dictionary
    3) Pass the updated hyperparameter dictionary to the training routine
    4) Output the loss on the validation dataset which acts as the metric for
       Bayesian optimization
This is then passed to gp_minimize() which conducts the optimization.

Once the n_calls number of iterations are complete, the function
output_results() is called to export the information of the hyperparameter
optimization procedure.

Inputs:
    - hyperp: dictionary storing set hyperparameter values
    - options: dictionary storing the set options
    - filepaths: instance of the FilePaths class storing the default strings for
                 importing and exporting required objects. This is to be updated
                 each time a new set of hyperparameter values is specified
    - n_calls: number of calls of the objective functional for optimization
    - space: skopt object storing the current hyperparameter values to be used
    - hyperp_of_interest_dict: dictionary of the hyperparameters to be searched
                               and their ranges
    - FilePathsClass: class storing the strings for importing and exporting
                      required objects. This class is required here for updating
                      the instance filepaths each time a new set of
                      hyperparameter values is specified.
    - project_paths: instance of the FilePathsProject class containing
                     project specific strings

Author: Hwan Goh, Oden Institute, Austin, Texas 2020
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
