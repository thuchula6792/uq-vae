#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:59:31 2019

@author: Jon Wittmer
"""

from mpi4py import MPI
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
def schedule_runs(scenarios, active_procs, proc_to_gpu_mapping, comm):
    available_processes = active_procs

    scenarios_left = len(scenarios)
    print(str(scenarios_left) + ' total runs left')

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

        if len(available_processes) > 0 and len(scenarios) > 0:
            curr_process = available_processes.pop(0) # rank of the process to send to
            curr_scenario = scenarios.pop(0)
            curr_scenario['gpu'] = str(proc_to_gpu_mapping[str(curr_process)]) # which GPUs we want to run the process on.

            # block here to make sure the process starts before moving on so we don't overwrite buffer
            print('current process: ' + str(curr_process))
            req = comm.isend(curr_scenario, curr_process, flags.NEW_RUN) # master process sending out new run
            req.wait() # without this, the message sent by comm.isend might get lost when this process hasn't been probed. With this, it essentially continues to message until its probe

        elif len(available_processes) > 0 and len(scenarios) == 0:
            while len(available_processes) > 0:
                # remove all leftover processes in the event that all scenarios are complete
                proc = available_processes.pop(0)
                comm.send([], proc, flags.EXIT)

        # rest for 7 seconds because there is no need to loop too fast
        # 7 seems like a good number.
        sleep(20)
