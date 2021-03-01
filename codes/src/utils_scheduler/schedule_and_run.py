#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:59:31 2019

@author: Jon Wittmer
"""

from mpi4py import MPI
import pynvml
from time import sleep
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class FLAGS:
    RECEIVED = 1
    RUN_FINISHED = 2
    EXIT = 3
    NEW_RUN = 4

###############################################################################
#                            Schedule and Run                                 #
###############################################################################
def schedule_runs(scenarios, nprocs, comm):

    pynvml.nvmlInit()

    scenarios_left = len(scenarios)
    print(str(scenarios_left) + ' total runs left')

    # initialize available processes
    available_processes = list(range(1, nprocs))

    flags = FLAGS()

    # start running tasks
    while scenarios_left > 0:

        # check worker processes for returning processes
        s = MPI.Status()
        comm.Iprobe(status=s)
        if s.tag == flags.RUN_FINISHED:
            print('Run ended. Starting new thread.')
            data = comm.recv(source=s.source, tag=s.tag)
            scenarios_left -= 1
            if len(scenarios) == 0:
                comm.send([], s.source, flags.EXIT)
            else:
                available_processes.append(s.source)

        # assign training to process
        available_gpus = available_GPUs() # check which GPUs have available memory or computation space

        if len(available_gpus) > 0 and len(available_processes) > 0 and len(scenarios) > 0:
            curr_process = available_processes.pop(0) # rank of the process to send to
            curr_scenario = scenarios.pop(0)
            curr_scenario['gpu'] = str(available_gpus.pop(0)) # which GPU we want to run the process on. Note that the extra "gpu" field is created here as well

            # block here to make sure the process starts before moving on so we don't overwrite buffer
            print('current process: ' + str(curr_process))
            req = comm.isend(curr_scenario, curr_process, flags.NEW_RUN) # master process sending out new run
            req.wait() # without this, the message sent by comm.isend might get lost when this process hasn't been probed. With this, it essentially continues to message until its probe

        elif len(available_processes) > 0 and len(scenarios) == 0:
            while len(available_processes) > 0:
                proc = available_processes.pop(0) # removes all leftover processes in the event that all scenarios are complete
                comm.send([], proc, flags.EXIT)

        sleep(60) # Tensorflow environment takes a while to fill up the GPU. This sleep command gives tensorflow time to fill up the GPU before checking if its available

def available_GPUs():
    available_gpus = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle  = pynvml.nvmlDeviceGetHandleByIndex(i)
        res     = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_res = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # Sometimes we want multiple processes on one GPU. This heuristic gives
        # an indication on whether the GPU has memory to support another process.
        # Adjust memory_threshold higher if less memory is needed for new process.
        memory_threshold = 30
        if res.gpu < memory_threshold and (mem_res.used / mem_res.total *100) < memory_threshold:
            available_gpus.append(i)
    return available_gpus
