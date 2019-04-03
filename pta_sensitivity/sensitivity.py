# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np
import itertools as it

## Some constants
yr_sec = 365.25*24*3600
fyr = 1/yr_sec

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

def transmission(designmatrix, toas, N=None,
                 nf=200, fmin=None, fmax=2e-7, freqs=None,
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
    if not from_G and N is None:
        err_msg = 'Covariance Matrix must be provided if constructing'
        err_msg += ' from R-matrix.'
        raise ValueError(err_msg)

    M = designmatrix

    ## Prep Correlation
    t1, t2 = np.meshgrid(toas, toas)
    tm = np.abs(t1-t2)
    T = toas.max()-toas.min()

    # make filter
    f0 = 1 / T
    if freqs is None:
        if fmin is None:
            fmin = f0/5
        ff = np.logspace(np.log10(fmin), np.log10(fmax), nf,dtype='float64')
        if exact_astro_freqs:
            ff = np.sort(np.append(ff,[fyr,2*fyr]))
            nf +=2
    else:
        ff = freqs

    Tmat = np.zeros(nf, dtype='float64')
    if from_G:
        G = G_matrix(M)
        m = G.shape[1]
        for ct, f in enumerate(ff):
            Tmat[ct] = np.real(np.sum(np.exp(-1j*2*np.pi*f*tm)
                                      *np.matmul(G,G.T))/m)
    else:
        R = R_matrix(M, N)
        N_TOA = M.shape[0]
        for ct, f in enumerate(ff):
            Tmat[ct] = np.real(np.sum(np.exp(-1j*2*np.pi*f*tm)*R)/N_TOA)

    return np.real(Tmat), ff, T


def response(freqs):
    """Timing residual response function."""
    return 1/(12*np.pi*freqs**2)

class sensitivity_curve(object):
    """
    Class for constructing PTA sensitivity curves.
    """
    def __init__(self, nf, Tspan, fmin=None, fmax=1e-7):
        self.nf = nf
        self.Tspan = Tspan*yr_sec
        f0 = 1 / Tspan
        if fmin is None:
            fmin = f0/5

        self.freqs = np.logspace(np.log10(fmin),np.log10(fmax),nf)

    def S_n(self):
        """Strain power sensitivity. """
        return None

    def h_c(self):
        """Characteristic strain sensitivity"""
        return None

    def Omega_gw(self):
        """Energy Density sensitivity"""
        return None

    def psd_postfit(self):
        """Postfit Residual Power"""
        return None

    def psd_prefit(self):
        """Prefit Residual Power"""
        return None

    def set_psr_sky_location(self,lat,long):
        """Set Latitude and Longitude of pulsars"""
        self._lat = lat
        self._long = long

    def calc_trans():
        """Calculate transmission functions for pulsars."""

    def calc_power():
        """Calculate """

    def create_white_noise_pow(self, sigma=None, dt=None, WN=None):
        if WN is None and np.logical_and(sigma is not None, dt is not None):
            self._white_noise = 2.0 * dt * (sigma)**2
            self._white_noise *= np.ones_like(self.freqs)
        elif WN is not None and np.logical_and(sigma is None, dt is None):
            self._white_noise = WN

        return self._white_noise

    def create_red_noise_pow(self, A=None, gamma=None, RN=None):
        if RN is None and np.logical_and(A is not None, gamma is not None):
            ff = self.freqs
            self._red_noise = A**2*(ff/fyr)**(-gamma)/(12*np.pi**2)
            self._red_noise *= yr_sec**3

        elif RN is not None and np.logical_and(A is None, gamma is None):
            self._red_noise = RN

        return self._red_noise

    def create_pulsar_term_noise_pow(self, Agw=None, gamma=None, PTN=None):
        if PTN is None and np.logical_and(Agw is not None, gamma is not None):
            ff = self.freqs
            self._pulsar_term_noise = Agw**2*(ff/fyr)**(-gamma)/(12*np.pi**2)
            self._pulsar_term_noise *= yr_sec**3

        elif PTN is not None and np.logical_and(Agw is None, gamma is None):
            self._pulsar_term_noise = PTN

        return self._pulsar_term_noise

def HellingsDownsCoeff(phi, theta):
    """
    Calculate Hellings and Downs coefficients from two lists of sky positions.

    Parameters
    ----------

    phi : array, list
        Pulsar axial coordinate.

    theta: array, list
        Pulsar azimuthal coordinate.

    Returns
    -------

    ThetaIJ : array
        Array of angles between pairs of pulsars.

    alphaIJ : array
        Array of Hellings and Downs relation coefficients.

    pairs: array
        A 2xNpair array of pair indices corresponding to input order of sky
        coordinates.

    alphaRSS : float
        Root-sum-squared value of all Hellings-Downs coefficients.

    """

    Npsrs = len(phi)
    # Npairs = np.int(Npsrs * (Npsrs-1) / 2.)
    psr_idx = np.arange(Npsrs)
    pairs = list(it.combinations(psr_idx,2))
    first, second = list(map(list, zip(*pairs)))
    cosThetaIJ = np.cos(theta[first]) * np.cos(theta[second]) \
                    + np.sin(theta[first]) * np.sin(theta[second]) \
                    * np.cos(phi[first] - phi[second])
    X = (1. - cosThetaIJ) / 2.
    alphaIJ = [1.5*x*np.log(x) - 0.25*x/4. + 0.5 if x!=0 else 1. for x in X]
    alphaIJ = np.array(alphaIJ)

    # calculate rss (root-sum-squared) of hellings-downs factor
    alphaRSS = np.sqrt(np.sum(alphaIJ**2))
    return np.arccos(cosThetaIJ), alphaIJ, np.array([first,second]), alphaRSS
