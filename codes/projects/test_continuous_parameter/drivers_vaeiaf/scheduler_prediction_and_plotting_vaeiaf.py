#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Schedules hyperparameter scenarios and drives training

You will need to specify:
    - In generate_scenarios_list() the set of hyperparameter scenarios you
      wish to use
    - In subprocess.Popen() whether the parameter-to-observable map is
      modelled or learned through specification of which training
      driver to call

Author: Jonathan Wittmer, Oden Institute, Austin, Texas 2019
'''
import os
import sys
sys.path.insert(0, os.path.realpath('../../../src'))
import json
import subprocess

from utils_scheduler.get_hyperparameter_combinations import get_hyperparameter_combinations

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                           Generate Scenarios List                           #
###############################################################################
def generate_scenarios_list():
    hyperp = {}
    hyperp['num_hidden_layers_encoder'] = [5]
    hyperp['num_hidden_layers_decoder'] = [2]
    hyperp['num_hidden_nodes_encoder']  = [500]
    hyperp['num_hidden_nodes_decoder']  = [500]
    hyperp['activation']                = ['relu']
    hyperp['num_iaf_transforms']        = [10]
    hyperp['num_hidden_nodes_iaf']      = [100]
    hyperp['activation_iaf']            = ['relu']
    hyperp['penalty_js']                = [0.00001, 0.001, 0.1, 0.5, 0.99]
    hyperp['num_data_train']            = [500, 1000, 2500, 5000]
    hyperp['batch_size']                = [100]
    hyperp['num_epochs']                = [1000]

    return get_hyperparameter_combinations(hyperp)

###############################################################################
#                                   Executor                                  #
###############################################################################
if __name__ == '__main__':

    #=== Get list of dictionaries of all combinations ===#
    scenarios_list = generate_scenarios_list()
    print('scenarios_list generated')

    #=== Convert dictionary to json string ===#
    for scenario in scenarios_list:
        scenario_json = json.dumps(scenario)
        proc = subprocess.Popen(
                ['./prediction_and_plotting_vaeiaf.py', f'{scenario_json}'])
        proc.wait()

    print('All scenarios computed')
