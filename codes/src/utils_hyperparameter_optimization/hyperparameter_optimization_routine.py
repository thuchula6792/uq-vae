#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  28 10:16:28 2020

@author: hwan
"""
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
                             training_routine, loss_val_index,
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
        training_routine(hyperp, options, filepaths,
                         data_dict, prior_dict)

        #=== Loading Metrics For Output ===#
        print('Loading Metrics')
        df_metrics = pd.read_csv(filepaths.trained_NN + "_metrics" + '.csv')
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
    #=== Assigning hyperp with Optimal Hyperparameters ===#
    for num, key in enumerate(hyperp_of_interest_dict.keys()):
        hyperp[key] = hyperp_opt_result.x[num]

    #=== Updating File Paths with Optimal Hyperparameters ===#
    filepaths = FilePathsClass(hyperp, options, project_paths)

    #=== Deleting Suboptimal Neural Networks ===#
    directories_list_trained_NN = os.listdir(
            path=filepaths.directory_hyperp_opt_trained_NN_case)

    for filename in directories_list_trained_NN:
        if filename != filepaths.NN_name:
            shutil.rmtree(filepaths.directory_hyperp_opt_trained_NN_case + '/' + filename)
            shutil.rmtree(filepaths.directory_hyperp_opt_tensorboard_case + '/' + filename)

    print('Suboptimal Trained Networks Deleted')
