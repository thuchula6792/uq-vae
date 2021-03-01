import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_bivariate_gaussian(filepath, mean, cov,
                            fig_size, lim_min, lim_max, colorbar_limits,
                            title, xlabel, ylabel):
    #=== Our 2-dimensional distribution will be over variables X and Y ===#
    x_axis = np.linspace(lim_min, lim_max, 100)
    y_axis = np.linspace(lim_min, lim_max, 100)
    X, Y = np.meshgrid(x_axis, y_axis)

    #=== Pack X and Y into a single 3-dimensional array ===#
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    #=== The distribution on the variables X, Y packed into pos ===#
    Z = multivariate_gaussian(pos, mean.flatten(), cov)

    #=== Plot ===#
    fig_contour, ax = plt.subplots(1,1)
    v = np.linspace(colorbar_limits[0], colorbar_limits[1], 40, endpoint=True)
    cp = ax.contourf(X, Y, Z, v)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cp, cax = cax)

    #=== Save figure ===#
    plt.savefig(filepath, dpi=100, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig_contour)

def multivariate_gaussian(pos, mean, cov):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """
    n = mean.shape[0]
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    N = np.sqrt((2*np.pi)**n * cov_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mean, cov_inv, pos-mean)

    return np.exp(-fac / 2) / N
