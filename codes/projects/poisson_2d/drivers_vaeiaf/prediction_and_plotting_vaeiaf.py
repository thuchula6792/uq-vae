#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:41:12 2019
@author: hwan
"""
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
from utils_project.prediction_and_plotting_routine_vaeiaf\
        import predict_and_plot, plot_and_save_metrics

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 Add Options                                 #
###############################################################################
def add_options(options):

    options.model_aware = True
    options.model_augmented = False

    return options

###############################################################################
#                                   Driver                                    #
###############################################################################
if __name__ == "__main__":

    #=== Hyperparameters ===#
    with open('../config_files/hyperparameters_vaeiaf.yaml') as f:
        hyperp = yaml.safe_load(f)
    if len(sys.argv) > 1: # if run from scheduler
        hyperp = command_line_json_string_to_dict(sys.argv[1], hyperp)
    hyperp = AttrDict(hyperp)

    #=== Options ===#
    with open('../config_files/options_vaeiaf.yaml') as f:
        options = yaml.load(f, Loader=yaml.FullLoader)
    options = AttrDict(options)
    options = add_options(options)
    options.posterior_diagonal_covariance = False
    options.posterior_iaf = True

    #=== File Names ===#
    project_paths = FilePathsProject(options)
    filepaths = FilePathsPredictionAndPlotting(hyperp, options, project_paths)

    #=== Predict and Save ===#
    predict_and_plot(hyperp, options, filepaths)

    #=== Plot and Save ===#
    plot_and_save_metrics(hyperp, options, filepaths)
