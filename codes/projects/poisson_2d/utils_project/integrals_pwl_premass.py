import numpy as np

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def integrals_pwl_premass(vertices_coords, position):
    """
    pwlIntpreMass computes the integrals of the mass matrix, without
    consideration of the parameter, for the nodes corresponding to the jth
    element. A piecewise linear approximation of the parameter is assumed.

    Inputs:
        vertices_coords - a 3 by 2 matrix where each row contains coordinates of the
            vertices for the jth element
        position - finds whether i is the first, second or third listed node of a
                   row of the Elements matrix

    Outputs:
        partialEMass - the 3 by 3 matrix which represents the entries of the
                    matrix delMass/delmu_(a_i)

    Hwan Goh 28/10/2013, University of Auckland, New Zealand
    Hwan Goh 1/7/2020, Transcribed to Python
    """

    L = np.array([[-1,1,0],[-1,0,1]]);
    Jac_T = np.matmul(L, vertices_coords); # The transpose of the Jacobian
    detJac = np.abs(np.linalg.det(Jac_T)); # Determinant of the Jacobian
    A = np.zeros((3,3));

    if position == 0:
        A[0,0] = 1/20;
        A[0,1] = 1/60;
        A[0,2] = 1/60;

        A[1,0] = 1/60;
        A[1,1] = 1/60;
        A[1,2] = 1/120;

        A[2,0] = 1/60;
        A[2,1] = 1/120;
        A[2,2] = 1/60;

    if position == 1:
        A[0,0] = 1/60;
        A[0,1] = 1/60;
        A[0,2] = 1/120;

        A[1,0] = 1/60;
        A[1,1] = 1/20;
        A[1,2] = 1/60;

        A[2,0] = 1/120;
        A[2,1] = 1/60;
        A[2,2] = 1/60;

    if position == 2:
        A[0,0] = 1/60;
        A[0,1] = 1/120;
        A[0,2] = 1/60;

        A[1,0] = 1/120;
        A[1,1] = 1/60;
        A[1,2] = 1/60;

        A[2,0] = 1/60;
        A[2,1] = 1/60;
        A[2,2] = 1/20;

    return detJac*A
