import tensorflow as tf
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def load_fem_matrices_tf(options, filepaths):

    #=== Forward Matrix and Mass Matrix ===#
    df_forward_matrix = pd.read_csv(filepaths.project.forward_matrix + '.csv')
    forward_matrix = df_forward_matrix.to_numpy()
    df_mass_matrix = pd.read_csv(filepaths.project.mass_matrix + '.csv')
    mass_matrix = df_mass_matrix.to_numpy()

    forward_matrix =\
            forward_matrix.reshape((options.parameter_dimensions, options.parameter_dimensions))
    mass_matrix =\
            mass_matrix.reshape((options.parameter_dimensions, options.parameter_dimensions))

    return tf.cast(forward_matrix, tf.float32), tf.cast(mass_matrix, tf.float32)
