import numpy as np
import tensorflow as tf

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def convert_dense_to_sparse_tensor(array):
    idx = np.where(array != 0.0)

    return tf.SparseTensor(np.vstack(idx).T, array[idx].astype(np.float32), array.shape)
