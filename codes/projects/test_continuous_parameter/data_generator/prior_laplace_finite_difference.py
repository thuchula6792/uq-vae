import numpy as np

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def prior_laplace_finite_difference(x):

    #=== Construct Finite Difference Matrix for [u_2, ... ,u_N-1] ===#
    h = (x[1]-x[0])**2
    num_dof = len(x)
    main_diag = 2*np.ones(num_dof-2)
    off_diag = -1*np.ones(num_dof-3)
    A = np.diag(main_diag) + np.diag(off_diag,1) + np.diag(off_diag,-1)
    A *= 1/h
    A = np.linalg.inv(A)

    #=== Construct Prior Matrices [u_2, ... ,u_N-1] ===#
    prior_covariance = A
    prior_covariance_cholesky = np.linalg.cholesky(prior_covariance)
    prior_covariance_cholesky_inverse = np.linalg.inv(prior_covariance_cholesky)

    #=== Implement Homogeneous Dirichlet BCs ===#
    prior_covariance = np.pad(
            prior_covariance, 1, 'constant', constant_values=(0))
    prior_covariance_cholesky = np.pad(
            prior_covariance_cholesky, 1, 'constant', constant_values=(0))
    prior_covariance_cholesky_inverse = np.pad(
            prior_covariance_cholesky_inverse, 1, 'constant', constant_values=(0))

    return prior_covariance,\
            prior_covariance_cholesky,\
            prior_covariance_cholesky_inverse
