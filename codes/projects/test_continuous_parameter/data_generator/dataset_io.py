import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def save_forward_vector(filepaths, forward_vector):
    #=== Save forward vector ===#
    df_forward_vector = pd.DataFrame({'forward_vector': forward_vector.flatten()})
    df_forward_vector.to_csv(filepaths.forward_vector + '.csv', index=False)

def save_parameter(filepaths, parameter):
    #=== Save parameter ===#
    df_parameter = pd.DataFrame({'parameter': parameter.flatten()})
    df_parameter.to_csv(filepaths.parameter + '.csv', index=False)

def save_state(filepaths, obs_indices, state, state_obs):
    #=== Save full state ===#
    df_state = pd.DataFrame({'state': state.flatten()})
    df_state.to_csv(filepaths.state_full + '.csv', index=False)

    #=== Save observation indices and data ===#
    df_obs_indices = pd.DataFrame({'obs_indices': obs_indices})
    df_obs_indices.to_csv(filepaths.obs_indices + '.csv', index=False)
    df_state_obs = pd.DataFrame({'state_obs': state_obs.flatten()})
    df_state_obs.to_csv(filepaths.state_obs + '.csv', index=False)

def load_parameter(filepaths, parameter_dimensions, num_data):

    df_parameters = pd.read_csv(filepaths.parameter + '.csv')
    parameters = df_parameters.to_numpy()
    parameters = parameters.reshape((num_data, parameter_dimensions))

    print('parameters loaded')

    return parameters
