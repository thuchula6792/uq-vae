import numpy as np
from numpy.random import multivariate_normal

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off

def plot_bivariate_gaussian(filepath, data_1, data_2, title, xlabel, ylabel):

    fig_contour = plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist2d(data_1, data_2, bins=100)
    fig_contour.tight_layout()
    plt.savefig(filepath)
    plt.close()
