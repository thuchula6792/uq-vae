import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_system_matrices(filepaths, options, mesh):
    #=== Construct Matrix ===#
    if options.continuous_linear_sin == True:
        forward_vector = np.sin(np.pi*mesh)
    if options.continuous_linear_uniform == True:
        forward_vector = np.random.uniform(2, 10, options.mesh_dimensions)

    #=== Save Forward Operator ===#
    df_forward_vector = pd.DataFrame({'forward_vector':forward_vector.flatten()})
    df_forward_vector.to_csv(filepaths.forward_vector + '.csv', index=False)

def load_system_matrices(options, filepaths):
    #=== Load Spatial Operator ===#
    df_forward_matrix = pd.read_csv(filepaths.forward_vector + '.csv')
    forward_vector = df_forward_matrix.to_numpy()

    return forward_vector
