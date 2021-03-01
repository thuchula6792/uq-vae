#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 9 22:03:02 2020

@author: hwan
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.ioff() # Turn interactive plotting off
import matplotlib.tri as tri

from utils_fenics.convert_array_to_dolfin_function import\
        convert_array_to_dolfin_function

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_cross_section(function_space,
                       parameter, mean, cov,
                       x_axis_limits, cross_section_y,
                       title,
                       filepath,
                       y_axis_limits):

    #=== Convert array to dolfin function ===#
    parameter_fe = convert_array_to_dolfin_function(function_space, parameter)
    mean_fe = convert_array_to_dolfin_function(function_space, mean)
    cov_fe = convert_array_to_dolfin_function(function_space, cov)

    #=== Extract mesh and triangulate ===#
    mesh = parameter_fe.function_space().mesh()
    coords = mesh.coordinates()
    elements = mesh.cells()
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], elements)

    #=== Linear Interpolators ===#
    interp_parameter = tri.LinearTriInterpolator(triangulation, parameter.flatten())
    interp_mean = tri.LinearTriInterpolator(triangulation, mean.flatten())
    interp_cov = tri.LinearTriInterpolator(triangulation, cov.flatten())

    #=== Interpolate values of cross section ===#
    x = np.linspace(x_axis_limits[0], x_axis_limits[1], 100, endpoint=True)
    parameter_cross = np.zeros(len(x))
    mean_cross = np.zeros(len(x))
    std_cross = np.zeros(len(x))
    for i in range(0,len(x)):
        parameter_cross[i] = interp_parameter(x[i], cross_section_y)
        mean_cross[i] = interp_mean(x[i], cross_section_y)
        std_cross[i] = np.exp(interp_cov(x[i], cross_section_y)/2)

    #=== Plotting ===#
    plt.plot(x, parameter_cross, 'r-', label='True Parameter')
    plt.plot(x, mean_cross, 'k-', label='Posterior Mean')
    plt.fill_between(x, mean_cross - 3*std_cross, mean_cross + 3*std_cross)
    plt.xlim(x_axis_limits)
    plt.ylim(y_axis_limits)
    plt.title(title)
    plt.legend(loc="upper left")
    plt.xlabel('x-coordinate')
    plt.ylabel('Parameter Value')

    #=== Save Figure ===#
    plt.savefig(filepath, dpi=100, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
