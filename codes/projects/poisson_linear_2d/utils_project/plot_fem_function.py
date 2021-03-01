# Obtained from:
# https://stackoverflow.com/questions/52202014/how-can-i-plot-2d-fem-results-using-matplotlib
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off
import matplotlib.tri as tri
import numpy as np

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_fem_function(file_path, plot_title, colorbar_max,
        nodes, elements,
        nodal_values):

    nodes_x = nodes[:,0]
    nodes_y = nodes[:,1]

    #=== Plot Mesh ===#
    # for element in elements:
    #     x = [nodes_x[element[i]] for i in range(len(element))]
    #     y = [nodes_y[element[i]] for i in range(len(element))]
    #     plt.fill(x, y, edgecolor='black', fill=False)

    #=== Triangulate Mesh ===#
    triangulation = tri.Triangulation(nodes_x, nodes_y, elements)

    #=== Plot FEM Function ====#
    v = np.linspace(0, colorbar_max, 15, endpoint=True)
    plt.tricontourf(triangulation, nodal_values.flatten(), v)
    plt.colorbar(ticks=v)
    plt.axis('equal')
    plt.savefig(file_path)
    plt.close()
