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
# from .skymap import SkySensitivity

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
    r'''
    Takes in a celestial longitude and lattitude and returns an `astropy.SkyCoord` object.
    
    Parameters
    ----------
    phi : float, array of floats
        The celestial longitude in solar system coordinates.
    theta : float, array of floats
        The celestial lattitude in solar system coordinates.
    
    Returns
    -------
    skycoord - astropy.SkyCoord object
        Can use this to convert to ra, dec, etc.
        (e.g. SkyCoord.ra.deg)

    '''

    return SkyCoord(phi*u.rad, ( theta - np.pi/2 )*u.rad)

def skycoord_to_Jname(skycoord):
    '''
    Takes in a SkyCoord object and returns the Jname of a pulsar with given coordinates.
    
    Parameters
    ----------
    skycoord - astropy.SkyCoord object
        Can use `theta_phi_to_SkyCoord()` to get this.
    
    Returns
    -------
    Jname - string, array of strings
        The traditional Jname of a pulsar with given coordinates
        (eg. 'J1713+0747')
    
    '''
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