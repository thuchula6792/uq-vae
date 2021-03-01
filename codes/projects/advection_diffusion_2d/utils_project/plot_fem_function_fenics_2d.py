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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.tri as tri

from utils_fenics.convert_array_to_dolfin_function import\
        convert_array_to_dolfin_function

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_fem_function_fenics_2d(function_space, nodal_values,
                                cross_section_y,
                                title, filepath,
                                fig_size, colorbar_limits,
                                plot_hline_flag):

    #=== Convert array to dolfin function ===#
    nodal_values_fe = convert_array_to_dolfin_function(function_space, nodal_values)

    #=== Extract mesh and triangulate ===#
    mesh = nodal_values_fe.function_space().mesh()
    coords = mesh.coordinates()
    elements = mesh.cells()
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], elements)

    #=== Plot figure ===#
    nodal_values = nodal_values_fe.compute_vertex_values(mesh)
    v = np.linspace(colorbar_limits[0], colorbar_limits[1], 40, endpoint=True)

    plt.figure(figsize = fig_size)
    ax = plt.gca()
    plt.title(title)
    figure = ax.tricontourf(triangulation, nodal_values, v, extend='max')
    if plot_hline_flag == True:
        plt.axhline(cross_section_y, color='k', linestyle='dashed', linewidth=3)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(figure, cax = cax)

    #=== Save figure ===#
    plt.savefig(filepath, dpi=100, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
