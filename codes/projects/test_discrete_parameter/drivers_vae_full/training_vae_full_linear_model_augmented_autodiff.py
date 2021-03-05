#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Drives training of a neural network under the uq-vae framework

The parameter-to-observable map is modelled and is required to be linear

In preparation for training the neural network, the code will:
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

Outputs will be stored in the following directories:
    - uq-vae/trained_nns/ for trained network and training metrics
    - uq-vae/tensorboard/ for Tensorboard training metrics

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
from utils_io.filepaths_vae import FilePathsTraining

# Import Project Utilities
from utils_project.filepaths_project import FilePathsProject
from utils_project.construct_data_dict import construct_data_dict
from utils_project.construct_prior_dict import construct_prior_dict
from utils_project.training_routine_vae_full_linear_model_augmented_autodiff\
        import training

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
    options.model_aware = False
    options.model_augmented = True
    options.posterior_full_covariance = True

    #=== File Paths ===#
    project_paths = FilePathsProject(options)
    filepaths = FilePathsTraining(hyperp, options, project_paths)

    #=== Data and Prior Dictionary ===#
    data_dict = construct_data_dict(hyperp, options, filepaths)
    prior_dict = construct_prior_dict(hyperp, options, filepaths,
                                      load_mean = True,
                                      load_covariance = False,
                                      load_covariance_inverse = True,
                                      load_covariance_cholesky = False,
                                      load_covariance_cholesky_inverse = False)

    #=== Initiate training ===#
    training(hyperp, options, filepaths,
             data_dict, prior_dict)
