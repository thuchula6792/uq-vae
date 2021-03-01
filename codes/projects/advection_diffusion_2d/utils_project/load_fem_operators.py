import numpy as np
import pandas as pd
from scipy import sparse

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def load_fem_operators(options, filepaths):

    if options.time_stepping_erk4 == True or options.time_stepping_lserk4 == True:
        #=== Load Spatial Operator ===#
        df_fem_operator_spatial = pd.read_csv(filepaths.project.fem_operator_spatial + '.csv')
        fem_operator_spatial = df_fem_operator_spatial.to_numpy()
        fem_operator_spatial = fem_operator_spatial.reshape(
                (options.parameter_dimensions, options.parameter_dimensions))
        fem_operator_spatial = fem_operator_spatial.astype(np.float32)
    else:
        fem_operator_spatial = 0

    if options.time_stepping_implicit == True:
        #=== Load Implicit Time Stepping Operators ===#
        df_fem_operator_implicit_ts = pd.read_csv(
                filepaths.project.fem_operator_implicit_ts + '.csv')
        fem_operator_implicit_ts = df_fem_operator_implicit_ts.to_numpy()
        fem_operator_implicit_ts = fem_operator_implicit_ts.reshape(
                (options.parameter_dimensions, options.parameter_dimensions))

        df_fem_operator_implicit_ts_rhs = pd.read_csv(
                filepaths.project.fem_operator_implicit_ts_rhs + '.csv')
        fem_operator_implicit_ts_rhs = df_fem_operator_implicit_ts_rhs.to_numpy()
        fem_operator_implicit_ts_rhs = fem_operator_implicit_ts_rhs.reshape(
                (options.parameter_dimensions, options.parameter_dimensions))
        fem_operator_implicit_ts = fem_operator_implicit_ts.astype(np.float32)
        fem_operator_implicit_ts_rhs = fem_operator_implicit_ts_rhs.astype(np.float32)
    else:
        fem_operator_implicit_ts = 0
        fem_operator_implicit_ts_rhs = 0

    return fem_operator_spatial, fem_operator_implicit_ts, fem_operator_implicit_ts_rhs
