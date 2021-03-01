#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:37:09 2020

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
from utils_io.filepaths_vae import FilePathsTraining

# Import Project Utilities
from utils_project.filepaths_project import FilePathsProject
from utils_project.construct_data_dict import construct_data_dict
from utils_project.construct_prior_dict import construct_prior_dict
from utils_project.training_routine_custom_vae_full_model_aware import trainer_custom

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 Add Options                                 #
###############################################################################
def add_options(options):

    #=== Use Distributed Strategy ===#
    options.distributed_training = False

    #=== Which GPUs to Use for Distributed Strategy ===#
    options.dist_which_gpus = '0,1,2,3'

    #=== Which Single GPU to Use ===#
    options.which_gpu = '2'

    return options

###############################################################################
#                                    Driver                                   #
###############################################################################
if __name__ == "__main__":

    #=== Hyperparameters ===#
    with open('../config_files/hyperparameters_vae_full.yaml') as f:
        hyperp = yaml.safe_load(f)
    if len(sys.argv) > 1:
        hyperp = command_line_json_string_to_dict(sys.argv[1], hyperp)
    hyperp = AttrDict(hyperp)

    #=== Options ===#
    with open('../config_files/options_vae_full.yaml') as f:
        options = yaml.safe_load(f)
    options = AttrDict(options)
    options = add_options(options)
    if len(sys.argv) > 1: # if run from scheduler
        options.which_gpu = sys.argv[2]
    options.model_aware = True
    options.model_augmented = False
    options.posterior_full_covariance = True

    #=== File Paths ===#
    project_paths = FilePathsProject(options)
    filepaths = FilePathsTraining(hyperp, options, project_paths)

    #=== Data and Prior Dictionary ===#
    data_dict = construct_data_dict(hyperp, options, filepaths)
    prior_dict = construct_prior_dict(hyperp, options, filepaths,
                                      load_mean = True,
                                      load_covariance = True,
                                      load_covariance_cholesky = False,
                                      load_covariance_cholesky_inverse = False)

    #=== Initiate training ===#
    trainer_custom(hyperp, options, filepaths,
                   data_dict, prior_dict)
