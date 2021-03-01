from mshr import *
import dolfin as dl

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_mesh(options):

    #=== Define Mesh ===#
    domain = Rectangle(dl.Point(0.0, 0.0),
                       dl.Point(options.domain_length, options.domain_width))

    if options.flow_navier_stokes == True:
        if options.hole_single_circle == True:
            geometry = domain\
                    - Circle(dl.Point(options.circle_center[0], options.circle_center[1]),
                                options.circle_radius, options.discretization_circle)
        if options.hole_two_rectangles == True:
            geometry = domain\
                        - Rectangle(dl.Point(options.rect_1_point_1[0], options.rect_1_point_1[1]),
                                    dl.Point(options.rect_1_point_2[0], options.rect_1_point_2[1]))\
                        - Rectangle(dl.Point(options.rect_2_point_1[0], options.rect_2_point_1[1]),
                                    dl.Point(options.rect_2_point_2[0], options.rect_2_point_2[1]))

    mesh = generate_mesh(geometry, options.discretization_domain)

    #=== Finite Element Space ===#
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1) # Function space for state, adjoint, and
                                               # parameter variables are chosen to be the same
    #=== Get the Mesh Topology ===#
    nodes = Vh.tabulate_dof_coordinates()
    dof = Vh.dim()

    return Vh, nodes, dof
