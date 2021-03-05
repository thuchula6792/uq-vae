import numpy as np
import pandas as pd


def save_prior(filepaths, prior_mean,
               prior_covariance, prior_covariance_inverse,
               prior_covariance_cholesky, prior_covariance_cholesky_inverse):

    #=== Mean ===#
    df_prior_mean = pd.DataFrame({'prior_mean': prior_mean.flatten()})
    df_prior_mean.to_csv(filepaths.prior_mean + '.csv', index=False)

    #=== Covariance ===#
    df_prior_covariance = pd.DataFrame({'prior_covariance': prior_covariance.flatten()})
    df_prior_covariance.to_csv(filepaths.prior_covariance + '.csv', index=False)

    #=== Covariance Inverse ===#
    df_prior_covariance_inverse = pd.DataFrame(
            {'prior_covariance_inverse': prior_covariance_inverse.flatten()})
    df_prior_covariance_inverse.to_csv(filepaths.prior_covariance_inverse + '.csv', index=False)

    #=== Covariance Cholesky ===#
    df_prior_covariance_cholesky = pd.DataFrame(
            {'prior_covariance_cholesky': prior_covariance_cholesky.flatten()})
    df_prior_covariance_cholesky.to_csv(
            filepaths.prior_covariance_cholesky + '.csv', index=False)

    #=== Covariance Cholesky Inverse ===#
    df_prior_covariance_cholesky_inverse = pd.DataFrame(
            {'prior_covariance_cholesky_inverse': prior_covariance_cholesky_inverse.flatten()})
    df_prior_covariance_cholesky_inverse.to_csv(
            filepaths.prior_covariance_cholesky_inverse + '.csv', index=False)

def load_prior(filepaths, dimensions):

    #=== Mean ===#
    df_prior_mean = pd.read_csv(filepaths.prior_mean + '.csv')
    prior_mean = df_prior_mean.to_numpy()

    #=== Covariance ===#
    df_prior_covariance = pd.read_csv(filepaths.prior_covariance + '.csv')
    prior_covariance = df_prior_covariance.to_numpy()
    prior_covariance = prior_covariance.reshape((dimensions, dimensions))

    #=== Covariance Inverse ===#
    df_prior_covariance_inverse =\
            pd.read_csv(filepaths.prior_covariance_inverse + '.csv')
    prior_covariance_inverse = df_prior_covariance_inverse.to_numpy()
    prior_covariance_inverse = prior_covariance_inverse.reshape((dimensions, dimensions))

    #=== Covariance Cholesky ===#
    df_prior_covariance_cholesky =\
            pd.read_csv(filepaths.prior_covariance_cholesky + '.csv')
    prior_covariance_cholesky = df_prior_covariance_cholesky.to_numpy()
    prior_covariance_cholesky = prior_covariance_cholesky.reshape((dimensions, dimensions))

    #=== Covariance Cholesky Inverse ===#
    df_covariance_cholesky_inverse =\
            pd.read_csv(filepaths.prior_covariance_cholesky_inverse + '.csv')
    prior_covariance_cholesky_inverse = df_covariance_cholesky_inverse.to_numpy()
    prior_covariance_cholesky_inverse =\
            prior_covariance_cholesky_inverse.reshape((dimensions, dimensions))

    return prior_mean, prior_covariance, prior_covariance_inverse,\
            prior_covariance_cholesky, prior_covariance_cholesky_inverse
