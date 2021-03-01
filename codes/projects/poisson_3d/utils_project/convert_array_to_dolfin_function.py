import numpy as np
from fenics import *

def convert_array_to_dolfin_function(V, nodal_values):
    nodal_values_dl = Function(V)
    nodal_values_dl.vector().set_local(np.squeeze(nodal_values))

    return nodal_values_dl
