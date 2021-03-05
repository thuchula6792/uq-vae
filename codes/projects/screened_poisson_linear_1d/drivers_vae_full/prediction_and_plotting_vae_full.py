#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Drives prediction and plotting using a neural network trained under the uq-vae framework

In preparation for prediction and plotting, the code will:
    1) Construct a dictionary containing the set hyperparameter values
       read from the .yaml file
    2) Construct a dictionary containing the set options
       read from the .yaml file
    3) Construct the project specific as well as neural-network related
       filepaths class from the hyperp and options dictionaries
    4) Construct a dictionary containing loaded training and testing data
       and related objects
    5) Construct a dictionary containing loaded prior related objects

You will need to specify:
    - In add_options:
        - Whether the neural network was trained using a modelled
          parameter-to-observable map or was the parameter-to-observable
          map learned

Outputs will be stored in uq-vae/figures/

Author: Hwan Goh, Oden Institute, Austin, Texas 2020
'''
import os
import sys
sys.path.insert(0, os.path.realpath('../../../src'))
sys.path.insert(0, os.path.realpath('..'))

import yaml
from attrdict import AttrDict

# Import src code
from utils_io.config_io import command_line_json_string_to_dict
from utils_io.filepaths_vae import FilePathsPredictionAndPlotting

# Import FilePaths class and plotting routine
from utils_project.filepaths_project import FilePathsProject
from utils_project.prediction_and_plotting_routine_vae_fenics import predict_and_plot
# from utils_project.prediction_and_plotting_movie_routine_vae_fenics import predict_and_plot
from utils_project.plot_and_save_metrics import plot_and_save_metrics

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 Add Options                                 #
###############################################################################
def add_options(options):

    options.model_aware = False
    options.model_augmented = True

    return options

###############################################################################
#                                   Driver                                    #
###############################################################################
if __name__ == "__main__":

    #=== Hyperparameters ===#
    with open('../config_files/hyperparameters_vae_full.yaml') as f:
        hyperp = yaml.safe_load(f)
    if len(sys.argv) > 1: # if run from scheduler
        hyperp = command_line_json_string_to_dict(sys.argv[1], hyperp)
    hyperp = AttrDict(hyperp)

    #=== Options ===#
    with open('../config_files/options_vae_full.yaml') as f:
        options = yaml.load(f, Loader=yaml.FullLoader)
    options = AttrDict(options)
    options = add_options(options)
    options.posterior_full_covariance = True

    #=== File Names ===#
    project_paths = FilePathsProject(options)
    filepaths = FilePathsPredictionAndPlotting(hyperp, options, project_paths)

    #=== Predict and Save ===#
    predict_and_plot(hyperp, options, filepaths)

    #=== Plot and Save ===#
    plot_and_save_metrics(hyperp, options, filepaths)
