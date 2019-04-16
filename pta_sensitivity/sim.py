# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np
from . import sensitivity as sens

__all__ = ['create_design_matrix',
           'sim_pta',
           ]

day_sec = 24*3600
yr_sec = 365.25*24*3600

def create_design_matrix(toas, RADEC=True, PROPER=False, PX=False):
    """
    Return designmatrix for quadratic spindown model + optional
    astrometric parameters

    :param toas: toa measurements [s]
    :param RADEC: (optional) Includes RA/DEC fitting
    :param PROPER: (optional) Includes proper motion fitting
    :param PX: (optional) Includes parallax fitting

    :return: M design matrix for QSD + optional astronometry

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
            designmatrix[:,ii] = np.sin(2*np.pi/3.16e7*toas)
        if model[ii] == 'DEC':
            designmatrix[:,ii] = np.cos(2*np.pi/3.16e7*toas)
        if model[ii] == 'PRA':
            designmatrix[:,ii] = toas*np.sin(2*np.pi/3.16e7*toas)
        if model[ii] == 'PDEC':
            designmatrix[:,ii] = toas*np.cos(2*np.pi/3.16e7*toas)
        if model[ii] == 'PX':
            designmatrix[:,ii] = np.cos(4*np.pi/3.16e7*toas)

    return designmatrix


def sim_pta(timespan, cad, sigma, phi, theta, Npsrs=None):
    """
    Make a simulated pulsar timing array with simple.

    Parameters
    ----------
    timespan : float, array, list
        Timespan of observations in [years].

    cad : float, array, list
        Cadence of observations [number/yr].

    sigma : float, array, list
        TOA RMS Error [sec]. Single float, Npsrs long array, or Npsrs x NTOA
        array excepted.

    phi : array, list
        Pulsars equatorial angles.

    theta : array, list
        Pulsars azimuthal angles.

    Return
    ------
    psrs : list
        List of `pta_sensitivity.Pulsar()` objects.

    """
    #Automatically deal with single floats and arrays.
    pars = [timespan, cad, sigma, phi, theta]
    keys = ['timespan', 'cad', 'sigma', 'phi', 'theta']
    haslen = [isinstance(par,(list,np.ndarray)) for par in pars]
    if any(haslen):
        L = [len(par) for par, hl in zip(pars, haslen) if hl]
        if not len(set(L))==1:
            raise ValueError('All arrays and lists must be the same length.')
        else:
            Npsrs = L[0]
    elif Npsrs is None:
        raise ValueError('If no array or lists are provided must set Npsrs!!')

    pars = [par * np.ones(Npsrs) if not hl else par
            for par, hl in zip(pars[:3], haslen[:3])]
    if all(haslen[3:]):
        pars.extend([phi,theta])
    else:
        raise ValueError('Must provide sky coordinates for all pulsars.')

    pars = dict(zip(keys,pars))

    psrs = []
    err_dim = pars['sigma'].ndim
    for ii in range(Npsrs):
        Ntoas = int(np.floor(pars['timespan'][ii]*pars['cad'][ii]))

        toas = np.linspace(0, pars['timespan'][ii]*yr_sec, Ntoas)
        if err_dim == 2:
            toaerrs = pars['sigma'][ii,:]
        else:
            toaerrs = pars['sigma'][ii]*np.ones(Ntoas)
        M = create_design_matrix(toas, RADEC=True, PROPER=True, PX=True)
        p = sens.Pulsar(toas, toaerrs, phi=pars['phi'][ii], theta=pars['theta'][ii],
                        N=np.diag(toaerrs**2))
        psrs.append(p)

    return psrs

def sim_SensitivityCurve():
    return None
