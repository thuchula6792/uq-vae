import numpy as np

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def integrals_pwl_prestiffness(vertices_coords):

    """
    integrals_pwl_prestuffness computes the integrals of the mass matrix, without
    consideration of the parameter, for the nodes corresponding to the jth
    element. A piecewise linear approximation of the parameter is assumed.

    Inputs:
        vertices_coords - a 3 by 2 matrix where each row contains coordinates of the
            vertices for the jth element
        pos - finds whether i is the first, second or third listed node of a
            row of the Elements matrix

    Outputs:
        partialEMass - the 3 by 3 matrix which represents the entries of the
                    matrix delMass/delmu_(a_i)

    Hwan Goh 28/10/2013, University of Auckland, New Zealand
    Hwan Goh 1/7/2020, Transcribed to Python
    """

    L = np.array([[-1,1,0],[-1,0,1]]);
    Jac_T = np.matmul(L,vertices_coords); # The transpose of the Jacobian
    detJac = np.abs(np.linalg.det(Jac_T)); # Determinant of the Jacobian
    S = np.matmul(np.linalg.inv(Jac_T),L) # The Inverse of the Jacobian multiplied by L

    return 1/6*np.matmul(S.T,S)*detJac
