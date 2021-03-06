#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Outputs information of Bayesian hyperparameter optimization

Following Bayesian optimization of select hyperparameters, this code will output
information on the procedure to uq-vae/hyperparameter_optimization/outputs/.
This includes:
    - convergence.png: convergence graph over the total number of calls
    - optimal_set_of_hyperparameters.txt: the value of the optimal
      hyperparameters for the hyperparameters specified in your search
    - scenarios_trained.txt: list of scenarios trained. Each entry contains the
                             value of the hyperparameter specified in your search
    - validation_losses.csv: list of the final validation loss for each of the
                              hyperparameter scenarios

Inputs:
    - hyperp_of_interest_dict: dictionary of the hyperparameters to be searched
                               and their ranges
    - hyperp_opt_result: training information for the network possessing the
                         optimal hyperparameter values

Author: Hwan Goh, Oden Institute, Austin, Texas 2020
'''
import os
import shutil

import numpy as np
import pandas as pd

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

# Import skopt code
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt import dump, load

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                               Output Results                                #
###############################################################################
def output_results(filepaths, hyperp_of_interest_dict, hyperp_opt_result):

    ##################################
    #   Display Optimal Parameters   #
    ##################################
    print('=================================================')
    print('      Hyperparameter Optimization Complete')
    print('=================================================')
    print('Optimized Validation Loss: {}\n'.format(hyperp_opt_result.fun))
    print('Optimized Parameters:')
    hyperp_of_interest_list = list(hyperp_of_interest_dict.keys())
    for n, parameter_name in enumerate(hyperp_of_interest_list):
        print(parameter_name + ': {}'.format(hyperp_opt_result.x[n]))

    #####################################
    #   Save Optimization Information   #
    #####################################
    #=== Creating Directory for Outputs ===#
    if not os.path.exists(filepaths.directory_hyperp_opt_outputs):
        os.makedirs(filepaths.directory_hyperp_opt_outputs)

    #=== Save .pkl File ===#
    dump(hyperp_opt_result, filepaths.hyperp_opt_skopt_res, store_objective=False)

    #=== Write Optimal Set Hyperparameters ===#
    with open(filepaths.hyperp_opt_optimal_parameters, 'w') as optimal_set_txt:
        optimal_set_txt.write('Optimized Validation Loss: {}\n'.format(hyperp_opt_result.fun))
        optimal_set_txt.write('\n')
        optimal_set_txt.write('Optimized parameters:\n')
        for n, parameter_name in enumerate(hyperp_of_interest_list):
            optimal_set_txt.write(parameter_name + ': {}\n'.format(hyperp_opt_result.x[n]))

    #=== Write List of Scenarios Trained ===#
    with open(filepaths.hyperp_opt_scenarios_trained, 'w') as scenarios_trained_txt:
        for scenario in hyperp_opt_result.x_iters:
            scenarios_trained_txt.write("%s\n" % scenario)

    #=== Write List of Validation Losses ===#
    validation_losses_dict = {}
    validation_losses_dict['validation_losses'] = hyperp_opt_result.func_vals
    df_validation_losses = pd.DataFrame(validation_losses_dict)
    df_validation_losses.to_csv(filepaths.hyperp_opt_validation_losses, index=False)

    #=== Convergence Plot ===#
    plot_convergence(hyperp_opt_result)
    plt.savefig(filepaths.hyperp_opt_convergence)

    print('Outputs Saved')
