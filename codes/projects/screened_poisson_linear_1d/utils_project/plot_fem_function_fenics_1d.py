#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:47:02 2019

@author: hwan
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off

from utils_fenics.convert_array_to_dolfin_function import\
        convert_array_to_dolfin_function

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_fem_function_fenics_1d(function_space, nodal_values,
                                title, filepath,
                                fig_size,
                                x_axis_limits, y_axis_limits):

    #=== Convert array to dolfin function ===#
    nodal_values_fe = convert_array_to_dolfin_function(function_space, nodal_values)

    #=== Extract mesh and triangulate ===#
    mesh = nodal_values_fe.function_space().mesh()
    coords = mesh.coordinates()
    elements = mesh.cells()

    #=== Nodal Values ===#
    nodal_values = nodal_values_fe.compute_vertex_values(mesh)

    #=== Plot figure ===#
    plt.figure(figsize = fig_size)
    ax = plt.gca()
    plt.title(title)
    plt.plot(coords, nodal_values, 'k-', label='Parameter')
    plt.xlim(x_axis_limits)
    plt.ylim(y_axis_limits)
    plt.legend(loc="upper left")
    plt.xlabel('x-coordinate')
    plt.ylabel('Parameter Value')

    #=== Save figure ===#
    plt.savefig(filepath, dpi=100, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
