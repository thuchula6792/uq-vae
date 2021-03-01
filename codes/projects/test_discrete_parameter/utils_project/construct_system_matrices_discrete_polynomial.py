import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_system_matrices(filepaths, options, mesh):
    #=== Construct Matrix ===#
    forward_matrix = np.ones((options.mesh_dimensions,1))
    grid_point_vec = np.expand_dims(mesh, axis=1)
    for d in range(1, options.parameter_dimensions):
        grid_point_vec = np.power(grid_point_vec, d)
        forward_matrix = np.concatenate(
                            (forward_matrix, grid_point_vec), axis=1)

    #=== Save Forward Operator ===#
    df_forward_matrix = pd.DataFrame({'forward_matrix':forward_matrix.flatten()})
    df_forward_matrix.to_csv(filepaths.forward_matrix + '.csv', index=False)

def load_system_matrices(options, filepaths):
    #=== Load Spatial Operator ===#
    df_forward_matrix = pd.read_csv(filepaths.forward_matrix + '.csv')
    forward_matrix = df_forward_matrix.to_numpy()

    return forward_matrix.reshape((options.mesh_dimensions, options.parameter_dimensions))
