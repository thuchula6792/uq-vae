import tensorflow as tf
import pandas as pd

from scipy import sparse

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

    #=== Load Vector ===#
    load_vector = sparse.load_npz(filepaths.project.load_vector + '.npz')
    load_vector = -options.load_vector_constant*load_vector
    load_vector = sparse.csr_matrix.todense(load_vector).T
    load_vector = tf.cast(load_vector, tf.float32)

    return tf.cast(forward_matrix, tf.float32),\
           tf.cast(mass_matrix, tf.float32),\
           load_vector
