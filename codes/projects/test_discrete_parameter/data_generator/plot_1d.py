import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_1d(values, filepath, dimensions,
            y_axis_min, y_axis_max,
            title, xlabel, ylabel):
    fig_parameter = plt.figure()
    x_axis = np.linspace(1, dimensions, dimensions, endpoint = True)
    plt.plot(x_axis, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(y_axis_min, y_axis_max)
    plt.savefig(filepath)
    plt.close(fig_parameter)
