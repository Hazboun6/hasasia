# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import scipy.optimize as sopt
import scipy.special as ssp
import scipy.integrate as si
import scipy.stats as ss
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as c

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
            designmatrix[:,ii] = toas**(ii) #Cute
        if model[ii] == 'RA':
            designmatrix[:,ii] = np.sin(2*np.pi/yr_sec*toas)
        if model[ii] == 'DEC':
            designmatrix[:,ii] = np.cos(2*np.pi/yr_sec*toas)
        if model[ii] == 'PRA':
            designmatrix[:,ii] = toas*np.sin(2*np.pi/yr_sec*toas)
        if model[ii] == 'PDEC':
            designmatrix[:,ii] = toas*np.cos(2*np.pi/yr_sec*toas)
        if model[ii] == 'PX':
            designmatrix[:,ii] = np.cos(4*np.pi/yr_sec*toas)

    return designmatrix

def fap(F, Npsrs=None):
    '''
    False alarm probability of the F-statistic
    Use None for the Fe statistic and the number of pulsars for the Fp stat.
    '''
    if Npsrs is None:
        N = [0,1]
    elif isinstance(Npsrs,int):
        N = np.arange((4*Npsrs)/2-1, dtype=float)
    # else:
    #     raise ValueError('Npsrs must be an integer or None (for Fe)')
    return np.exp(-F)*np.sum([(F**k)/np.math.factorial(k) for k in N])

def pdf_F_signal(F, snr, Npsrs=None):
    if Npsrs is None:
        N = 4
    elif isinstance(Npsrs,int):
        N = int(4 * Npsrs)
    return ss.ncx2.pdf(2*F, N, snr**2)

def false_dismissal_prob(F0, snr, Npsrs=None, iota_psi_ave=False):
    '''
    False dismissal probability of the F-statistic
    Use None for the Fe statistic and the number of pulsars for the Fp stat.
    '''
    if Npsrs is None:
        N = 4
    elif isinstance(Npsrs,int):
        N = int(4 * Npsrs)
    if iota_psi_ave:
        return ss.chi2.cdf(2*F0, df=N, loc=snr**2)
    else:
        return ss.ncx2.cdf(2*F0, df=N, nc=snr**2)

def detection_prob(F0, snr, Npsrs=None, iota_psi_ave=False):
    '''
    Detection probability of the F-statistic
    Use None for the Fe and the number of pulsars for the Fp stat.
    '''
    return 1 - false_dismissal_prob(F0, snr, Npsrs, iota_psi_ave)

def _solve_F_given_fap(fap0=0.003, Npsrs=None):
    return sopt.fsolve(lambda F :fap(F, Npsrs=Npsrs)-fap0, 10)

def _solve_F_given_fdp_snr(fdp0=0.05, snr=3, Npsrs=None, iota_psi_ave=False):
    Npsrs = 1 if Npsrs is None else Npsrs
    F0 = (4*Npsrs+snr**2)/2
    return sopt.fsolve(lambda F :false_dismissal_prob(F, snr, Npsrs=Npsrs, iota_psi_ave=iota_psi_ave)-fdp0, F0)

def _solve_snr_given_fdp_F(fdp0=0.05, F=3, Npsrs=None, iota_psi_ave=False):
    Npsrs = 1 if Npsrs is None else Npsrs
    snr0 = np.sqrt(2*F-4*Npsrs)
    return sopt.fsolve(lambda snr :false_dismissal_prob(F, snr, Npsrs=Npsrs, iota_psi_ave=iota_psi_ave)-fdp0, snr0)

def _solve_F0_given_SNR(snr=3, Npsrs=None):
    '''
    Returns the F0 (Fe stat threshold for a specified SNR)
    Use None for the Fe and the number of pulsars for the Fp stat.
    '''
    Npsrs = 1 if Npsrs is None else Npsrs 
    return 0.5*(4.*Npsrs+snr**2.)

def strain_and_chirp_mass_to_luminosity_distance(h, M_c, f0):
    r'''
    Returns the luminosity distance to a source given the strain, chirp mass, and GW frequency.
    
    Parameters
    ----------
    h : float
        The strain of the source.
    M_c : float
        The chirp mass of the source [Msun].
    f0 : float
        The GW frequency of the source [Hz].
        
    Returns
    -------
    D_L : float
        The luminosity distance to the source [Mpc].
    '''
    return (4*c.c / (h * u.m/u.m)
            * np.power(c.G * M_c * u.Msun/c.c**3, 5/3)
            * np.power(np.pi * f0 * u.Hz, 2/3)).to('Mpc')

def theta_phi_to_SkyCoord(theta, phi):
    """
    Parameters
    ==========
    phi : float, array of floats
        The celestial longitude in solar system coordinates.
    theta : float, array of floats
        The celestial lattitude in solar system coordinates.
    
    Returns
    =======
    skycoord - astropy.SkyCoord object
        Can use this to convert to ra, dec, etc.
        (e.g. SkyCoord.ra.deg)

    Converts an inputed longitude and lattitude into an `astropy.SkyCoord` object.

    """

    return SkyCoord(phi*u.rad, ( theta - np.pi/2 )*u.rad)

def skycoord_to_Jname(skycoord):
    """
    Parameters
    ==========
    skycoord - astropy.SkyCoord object
        Can use `theta_phi_to_SkyCoord()` to get this.
    
    Returns
    =======
    Jname - string, array of strings
        The traditional Jname of a pulsar with given coordinates
        (eg. 'J1713+0747')
    
    """
    coord_pieces = [
            str(skycoord.ra.hms[0]).split('.')[0], 
            str(skycoord.ra.hms[1]).split('.')[0], 
            str(abs(skycoord.dec.hms[0])).split('.')[0], 
            str(abs(skycoord.dec.hms[1])).split('.')[0]
            ]
    sign = ['-','+'][int(skycoord.dec.hms[0]>0)]
    for i, piece in enumerate(coord_pieces):
        if len(str(abs(int(piece)))) < 2:
            coord_pieces[i] = '0' + str(piece)
    return 'J' + coord_pieces[0] + coord_pieces[1] + sign + coord_pieces[2] +coord_pieces[3]