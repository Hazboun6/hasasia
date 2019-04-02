# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np

## Some constants
yr = 365.25*24*3600
fyr = 1/yr

def R_matrix(designmatrix, N):
    """
    Create R matrix as defined in Ellis et al (2013)
    and Demorest et al (2012)

    :param designmatrix: Design matrix
    :param err: TOA uncertainties [s]

    :return: R matrix

    """
    M = designmatrix
    if N.ndim ==1:
        Ninv = np.diag(1/N)
    else:
        Ninv = np.linalg.inv(N)

    MtWsqM = np.matmul(M.T,np.matmul(Ninv,M))
    MtWsqMinv = np.linalg.inv(MtWsqM)
    Id = np.eye(M.shape[0])

    return Id - np.matmul(np.matmul(M,np.matmul(MtWsqMinv,M.T)),Ninv)

def G_matrix(designmatrix):
    """
    Create G matrix as defined in van Haasteren 2013

    Parameters
    ----------

    designmatrix : array
        Design matrix for a pulsar timing model.

    Return
    ------
    G matrix

    """
    M = designmatrix
    n , m = M.shape
    U, _ , _ = np.linalg.svd(M, full_matrices=True)

    return U[:,m:]

def transmission(designmatrix, toas, N,
                 nf=200, fmin=None, fmax=2e-7,
                 exact_astro_freqs = False, from_G=False):
    """
    Calculate the transmission function for a given pulsar design matrix, TOAs
    and TOA errors.

    Parameters
    ----------

    designmatrix : array
        Design matrix for a pulsar timing model, N_TOA x N_param.

    toas : array
        Times-of-arrival for pulsar, N_TOA long.

    N : array
        Covariance matrix for pulsar time-of-arrivals, N_TOA x N_TOA. Often just
        a diagonal matrix of inverse TOA errors squared.

    nf : int, optional
        Number of frequencies at which to calculate transmission function.

    fmin : float, optional
        Minimum frequency at which to calculate transmission function.

    fmax: float, optional
        Maximum frequency at which to calculate transmission function.

    exact_astro_freqs: bool, optional
        Whether to use exact 1/year and 2/year frequency values in calculation.

    from_G: bool, optional
        Whether to use G matrix for transmission function calculate. Default is
        False, in which case R matrix is used.
    """

    M = designmatrix

    ## Prep Correlation
    t1, t2 = np.meshgrid(toas, toas)
    tm = np.abs(t1-t2)
    T = toas.max()-toas.min()

    # make filter
    f0 = 1 / T
    if fmin is None:
        fmin = f0/5
    ff = np.logspace(np.log10(fmin), np.log10(fmax), nf,dtype='float64')
    if exact_astro_freqs:
        ff = np.sort(np.append(ff,[fyr,2*fyr]))
        nf +=2

    Tmat = np.zeros(nf, dtype='float64')
    if from_G:
        G = G_matrix(M)
        m = G.shape[1]
        for ct, f in enumerate(ff):
            Tmat[ct] = np.real(np.sum(np.exp(-1j*2*np.pi*f*tm)*np.matmul(G,G.T))/m)
    else:
        R = R_matrix(M, N)
        N_TOA = M.shape[0]
        for ct, f in enumerate(ff):
            Tmat[ct] = np.real(np.sum(np.exp(-1j*2*np.pi*f*tm)*R)/N_TOA)

    return np.real(Tmat), ff, T


def response(freqs):
    """Timing residual response function."""
    return 1/(12*np.pi*freqs**2)
