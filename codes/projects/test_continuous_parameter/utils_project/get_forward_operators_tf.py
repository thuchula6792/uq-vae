import tensorflow as tf
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def load_forward_operator_tf(options, filepaths):

    df_forward_operator = pd.read_csv(filepaths.project.forward_operator + '.csv')
    forward_operator = df_forward_operator.to_numpy()
    return tf.cast(tf.transpose(forward_operator), tf.float32)
