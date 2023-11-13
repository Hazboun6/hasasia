# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import scipy.optimize as sopt
import scipy.special as ssp
import scipy.integrate as si
import scipy.stats as ss

__all__ = ['create_design_matrix',
           'fap',
           ]

day_sec = 24*3600
yr_sec = 365.25*24*3600

def create_design_matrix(toas, RADEC=False, PROPER=False, PX=False):
    """
    Return designmatrix for quadratic spindown model + optional
    astrometric parameters

    Parameters
    ----------
    toas : array
        TOA measurements [s]

    RADEC : bool, optional
        Includes RA/DEC fitting.

    PROPER : bool, optional
        Includes proper motion fitting.

    PX : bool, optional
        Includes parallax fitting.

    Returns
    -------
    M : array
        Design matrix for quadratic spin down + optional astrometry fit.

   """
    model = ['QSD', 'QSD', 'QSD']
    if RADEC:
        model.append('RA')
        model.append('DEC')
    if PROPER:
        model.append('PRA')
        model.append('PDEC')
    if PX:
        model.append('PX')

    ndim = len(model)
    designmatrix = np.zeros((len(toas), ndim))

    for ii in range(ndim):
        if model[ii] == 'QSD': #quadratic spin down fit
            designmatrix[:, ii] = toas**(ii) #Cute
        if model[ii] == 'RA':
            designmatrix[:, ii] = np.sin(2*np.pi / yr_sec * toas)
        if model[ii] == 'DEC':
            designmatrix[:, ii] = np.cos(2*np.pi / yr_sec * toas)
        if model[ii] == 'PRA':
            designmatrix[:, ii] = toas * np.sin(2*np.pi / yr_sec * toas)
        if model[ii] == 'PDEC':
            designmatrix[:, ii] = toas * np.cos(2*np.pi / yr_sec * toas)
        if model[ii] == 'PX':
            designmatrix[:, ii] = np.cos(4*np.pi / yr_sec * toas)

    return designmatrix

def fap(F, Npsrs=None):
    '''
    False alarm probability of the F-statistic
    Use None for the Fe statistic and the number of pulsars for the Fp stat.
    '''
    if Npsrs is None:
        N = [0, 1]
    elif isinstance(Npsrs, int):
        N = np.arange((4*Npsrs)/2 - 1, dtype=float)
    # else:
    #     raise ValueError('Npsrs must be an integer or None (for Fe)')
    return np.exp(-F)*np.sum([(F**k)/np.math.factorial(k) for k in N])

def pdf_F_signal(F, snr, Npsrs=None):
    if Npsrs is None:
        N = 4
    elif isinstance(Npsrs, int):
        N = int(4 * Npsrs)
    return ss.ncx2.pdf(2*F, N, snr**2)

def fdp(F0, snr, Npsrs=None, sky_ave=False):
    '''
    False detection probability of the F-statistic
    Use None for the Fe statistic and the number of pulsars for the Fp stat.
    '''
    if Npsrs is None:
        N = 4
    elif isinstance(Npsrs, int):
        N = int(4 * Npsrs)
    if sky_ave:
        return ss.chi2.cdf(2*F0, df=N, loc=snr**2)
    else:
        return ss.ncx2.cdf(2*F0, df=N, nc=snr**2)


def _solve_F_given_fap(fap0=0.003, Npsrs=None):
    return sopt.fsolve(lambda F :fap(F, Npsrs=Npsrs)-fap0, 10)

def _solve_F_given_fdp_snr(fdp0=0.05, snr=3, Npsrs=None, sky_ave=False):
    Npsrs = 1 if Npsrs is None else Npsrs
    F0 = (4*Npsrs + snr**2)/2
    return sopt.fsolve(lambda F :fdp(F, snr, Npsrs=Npsrs, sky_ave=sky_ave)-fdp0, F0)

def _solve_snr_given_fdp_F(fdp0=0.05, F=3, Npsrs=None, sky_ave=False):
    Npsrs = 1 if Npsrs is None else Npsrs
    snr0 = np.sqrt(2*F - 4*Npsrs)
    return sopt.fsolve(lambda snr :fdp(F, snr, Npsrs=Npsrs, sky_ave=sky_ave)-fdp0, snr0)
