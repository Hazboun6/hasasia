# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import scipy.optimize as sopt
import scipy.special as ssp
import scipy.integrate as si
import scipy.stats as ss
import scipy.constants as const # the same as enterprise.constants
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as c
# from .skymap import SkySensitivity

__all__ = ['create_design_matrix',
           'fap',
           'pdf_F_signal',
           '_solve_F0_given_SNR',
           '_solve_F_given_fap',
           'strain_and_chirp_mass_to_luminosity_distance',
           'char_strain_to_strain_amp',
           'theta_phi_to_SkyCoord',
           'skycoord_to_Jname',
           'distance_on_sphere',
           ]

day_sec = 24*3600
yr_sec = 365.25*24*3600

# these should match the definition in enterprise_extensions.chromatic.solar_wind
AU_light_sec = const.astronomical_unit / const.speed_of_light # 1 AU in light seconds
AU_pc = const.astronomical_unit / const.parsec  # 1 AU in parsecs (for DM normalization)

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
    """
    Probability density of the F-statistic with a signal present.
    
    F - float
        The F-statistic value.
    snr - float
        The signal-to-noise ratio of the signal.
    Npsrs - int, None
        Use None for the Fe statistic and the number of pulsars for the Fp stat.
    """
    if Npsrs is None:
        N = 4
    elif isinstance(Npsrs,int):
        N = int(4 * Npsrs)
    return ss.ncx2.pdf(2*F, N, snr**2)

def _solve_F_given_fap(fap0=0.003, Npsrs=None):
    return sopt.fsolve(lambda F :fap(F, Npsrs=Npsrs)-fap0, 10)

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

def char_strain_to_strain_amp(hc, fc, df):
    r'''
    Calculate the strain amplitude of single sources given
    their characteristic strains.

    Parameters
    ----------
    hc : array_like
        Characteristic strain of the single sources.
    fc : array_like
        Observed orbital frequency bin centers.
    df : array_like
        Observed orbital frequency bin widths.

    Returns
    -------
    hs : 
        Strain amplitude of the single sources.

    '''
    return hc * np.sqrt(df/fc)

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

def distance_on_sphere(lat1, lon1, lat2, lon2):
    ''''
    Returns the distance between two points on the surface of the unit sphere.
    
    Parameters
    ==========
    lat1, lat2 - float
        lattitude of first and second point respectively
    lon1, lon2 - float
        longitude of first and second point respectively
    Returns
    =======
    distance - float
        distance between two points
    '''
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula for distance
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = c
    
    return distance


######## solar wind utils  -- copied from enterprise_extensions.chromatic.solar_wind ########

def _dm_solar_close(n_earth, r_earth):
    return (n_earth * AU_light_sec * AU_pc / r_earth)


def _dm_solar(n_earth, theta, r_earth):
    return ((np.pi - theta) *
            (n_earth * AU_light_sec * AU_pc
             / (r_earth * np.sin(theta))))

def dm_solar(n_earth, theta, r_earth):
    """
    Calculates Dispersion measure due to 1/r^2 solar wind density model.
    ::param :n_earth Solar wind proto/electron density at Earth (1/cm^3)
    ::param :theta: angle between sun and line-of-sight to pulsar (rad)
    ::param :r_earth :distance from Earth to Sun in (light seconds).
    See You et al. 2007 for more details.
    """
    return np.where(np.pi - theta >= 1e-5,
                    _dm_solar(n_earth, theta, r_earth),
                    _dm_solar_close(n_earth, r_earth))

def theta_impact(planetssb, sunssb, pos_t):
    """
    Use the attributes of an enterprise Pulsar object to calculate the
    solar impact angle.

    ::param :planetssb Solar system barycenter time series supplied with
        enterprise.Pulsar objects.
    ::param :sunssb Solar system sun-to-barycenter timeseries supplied with
        enterprise.Pulsar objects.
    ::param :pos_t Unit vector to pulsar position over time in ecliptic
        coordinates. Supplied with enterprise.Pulsar objects.

    returns: Solar impact angle (rad), Distance to Earth (R_earth),
             impact distance (b), perpendicular distance (z_earth)
    """
    earth = planetssb[:, 2, :3]
    sun = sunssb[:, :3]
    earthsun = earth - sun
    R_earth = np.sqrt(np.einsum('ij,ij->i', earthsun, earthsun))
    Re_cos_theta_impact = np.einsum('ij,ij->i', earthsun, pos_t)

    theta_impact = np.arccos(-Re_cos_theta_impact / R_earth)
    b = np.sqrt(R_earth**2 - Re_cos_theta_impact**2)

    return theta_impact, R_earth, b, -Re_cos_theta_impact

def solar_wind_geometric_factor(radio_freqs, planetssb, sunssb, pos_t):
    """
    Calculate the geometric delay factor due to solar wind dispersion.
    Parameters
    ----------
    ::param :radio_freqs: radio frequencies in MHz
    ::param :planetssb Solar system barycenter time series supplied with
        enterprise.Pulsar objects.
    ::param :sunssb Solar system sun-to-barycenter timeseries supplied with
        enterprise.Pulsar objects.
    ::param :pos_t Unit vector to pulsar position over time in ecliptic
        coordinates. Supplied with enterprise.Pulsar objects.
    Returns
    -------
    ::return :dt_DM: Dispersive delay induced by solar wind. Probably dimensionless.
    """
    theta, R_earth, _, _ = theta_impact(planetssb, sunssb, pos_t)
    dm_sol_wind = dm_solar(1.0, theta, R_earth)
    dt_DM = dm_sol_wind * 4.148808e3 /(radio_freqs**2)

    return dt_DM