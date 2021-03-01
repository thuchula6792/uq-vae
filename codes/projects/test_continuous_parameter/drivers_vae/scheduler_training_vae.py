#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:53:06 2019

@author: Jon Wittmer
"""
import subprocess
from mpi4py import MPI

import os
import sys
sys.path.insert(0, os.path.realpath('../../../src'))
import json

from utils_scheduler.get_hyperparameter_combinations import get_hyperparameter_combinations
from utils_scheduler.schedule_and_run import schedule_runs

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class FLAGS:
    RECEIVED = 1
    RUN_FINISHED = 2
    EXIT = 3
    NEW_RUN = 4

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
    hyperp['penalty_js']                = [0.00001, 0.001, 0.1, 0.5, 0.99]
    hyperp['num_data_train']            = [500, 1000, 2500, 5000]
    hyperp['batch_size']                = [100]
    hyperp['num_epochs']                = [1000]

    return get_hyperparameter_combinations(hyperp)

###############################################################################
#                                   Executor                                  #
###############################################################################
if __name__ == '__main__':

    # To run this code "mpirun -n 5 ./scheduler_training_vae.py" in command line

    # mpi stuff
    comm   = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank   = comm.Get_rank()

    # By running "mpirun -n <number> ./scheduler_", each
    # process is cycled through by their rank
    if rank == 0: # This is the master processes' action
        # Generate scenarios list
        scenarios_list = generate_scenarios_list()

        # Schedule and run processes
        schedule_runs(scenarios_list, nprocs, comm)

    else:  # This is the worker processes' action
        while True:
            status = MPI.Status()
            scenario = comm.recv(source=0, status=status)

            if status.tag == FLAGS.EXIT:
                break

            # Dump scenario to driver code and run
            scenario_json = json.dumps(scenario)
            proc = subprocess.Popen(['./training_vae_model_aware.py',
                f'{scenario_json}',f'{scenario["gpu"]}'])
            # proc = subprocess.Popen(['./training_vae_model_augmented_autodiff.py',
                # f'{scenario_json}',f'{scenario["gpu"]}'])
            proc.wait()

            req = comm.isend([], 0, FLAGS.RUN_FINISHED)
            req.wait()

    print('All scenarios computed')
