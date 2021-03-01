import tensorflow as tf
import pandas as pd
from scipy import sparse
from utils_misc.convert_scipy_sparse_to_sparse_tensor import convert_scipy_sparse_to_sparse_tensor
from utils_misc.convert_dense_to_sparse_tensor import convert_dense_to_sparse_tensor

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def load_fem_matrices_tf(options, filepaths,
                         load_premass = 0,
                         load_prestiffness = 0):
    #=== Premass ===#
    if load_premass == 1:
        premass = sparse.load_npz(filepaths.project.premass + '.npz')
    else:
        premass = 0

    #=== Prestiffness ===#
    if load_prestiffness == 1:
        prestiffness = sparse.load_npz(filepaths.project.prestiffness + '.npz')
    else:
        prestiffness = 0

    #=== Boundary Matrix ===#
    boundary_matrix = sparse.load_npz(filepaths.project.boundary_matrix + '.npz')
    boundary_matrix = options.boundary_matrix_constant*boundary_matrix

    #=== Load Vector ===#
    load_vector = sparse.load_npz(filepaths.project.load_vector + '.npz')
    load_vector = -options.load_vector_constant*load_vector

    #=== Convert to Tensor ===#
    if load_premass == 1:
        premass = convert_scipy_sparse_to_sparse_tensor(premass)
        # premass = convert_dense_to_sparse_tensor(premass)
        # premass = tf.cast(premass, tf.float32)
    if load_prestiffness == 1:
        prestiffness = convert_scipy_sparse_to_sparse_tensor(prestiffness)
        # prestiffness = convert_dense_to_sparse_tensor(prestiffness)
        # prestiffness = tf.cast(prestiffness, tf.float32)

    #=== Convert to Dense ===# (no tensorflow support for sparse solve yet)
    boundary_matrix = sparse.csr_matrix.todense(boundary_matrix)
    load_vector = sparse.csr_matrix.todense(load_vector).T
    boundary_matrix = tf.cast(boundary_matrix, tf.float32)
    load_vector = tf.cast(load_vector, tf.float32)

    return premass, prestiffness, boundary_matrix, load_vector
