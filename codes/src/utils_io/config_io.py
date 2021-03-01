import os
import yaml
import json
import warnings

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def command_line_json_string_to_dict(json_string, hyperp):
    '''
    Overwrite the hyperparameters loaded from file.

    Note that the there is no checking to ensure that the command line
    arguments are in fact hyperparameters used by code, so spelling mistakes
    will cause the hyperparameters loaded from file to be used rather than
    the hyperparameter from command line. This was done on purpose to enable
    the capability to create new hyperparameters without changing hard-coding
    expected hyperparameters. This is more flexible, but more error prone.
    Warning added to notify user if key is not already present in hp dictionary.

    Assumes that all of the hyperparameters from the command line
    are in the form of a single JSON string in args[1]
    '''
    #=== Overwrite Hyperparameter Keys ===#
    command_line_arguments = json.loads(json_string)
    for key, value in command_line_arguments.items():
        if key not in hyperp:
            warnings.warn(
                f'Key "{key}" is not in hyperp and has been added. Make sure this is correct.')
        hyperp[key] = value
    return hyperp

def dump_attrdict_as_yaml(attrdict, filepath, filename):

    dump_dict = {}
    for key, value in attrdict.items():
        dump_dict[key] = value

    with open(filepath + '/' + filename + '.yaml', 'w') as f:
        yaml.dump(dump_dict, f)
