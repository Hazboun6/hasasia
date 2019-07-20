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


def sim_pta(timespan, cad, sigma, phi, theta, Npsrs=None,
            A_rn=None, alpha=None, freqs=None):
    """
    Make a simulated pulsar timing array. Using the available parameters,
    the function returns a list of pulsar objects encoding them.

    Parameters
    ----------
    timespan : float, array, list
        Timespan of observations in [years].

    cad : float, array, list
        Cadence of observations [number/yr].

    sigma : float, array, list
        TOA RMS Error [sec]. Single float, Npsrs long array,
        or Npsrs x NTOA array excepted.

    phi : array, list
        Pulsar's longitude in ecliptic coordinates.

    theta : array, list
        Pulsar's colatitude in ecliptic coordinates.

    Npsrs : int, optional
        Number of pulsars. Only needed if all pulsars have the same
        noise characteristics.

    A_rn : float, optional
        Red noise amplitude to be injected for each pulsar.

    alpha : float, optional
        Red noise spectral index to be injected for each pulsar.

    freqs : array, optional
        Array of frequencies at which to calculate the red noise. Same
        array used for all pulsars.

    Return
    ------
    psrs : list
        List of `hasasia.Pulsar()` objects.

    """
    #Automatically deal with single floats and arrays.
    if A_rn is None and alpha is None:
        pars = [timespan, cad, sigma, phi, theta]
        keys = ['timespan', 'cad', 'sigma', 'phi', 'theta']
        stop = 3
    elif None in [A_rn,alpha,freqs]:
        err_msg = 'A_rn, alpha and freqs must all be specified for '
        err_msg += 'in order to build C_rn.'
        raise ValueError(err_msg)
    else:
        pars = [timespan, cad, sigma, A_rn, alpha, phi, theta]
        keys = ['timespan', 'cad', 'sigma', 'A_rn', 'alpha',
                'phi', 'theta']
        stop = 5

    haslen = [isinstance(par,(list,np.ndarray)) for par in pars]
    if any(haslen):
        L = [len(par) for par, hl in zip(pars, haslen) if hl]
        if not len(set(L))==1:
            err_msg = 'All arrays and lists must be the same length.'
            raise ValueError(err_msg)
        else:
            Npsrs = L[0]
    elif Npsrs is None:
        err_msg = 'If no array or lists are provided must set Npsrs!!'
        raise ValueError(err_msg)

    pars = [par * np.ones(Npsrs) if not hl else par
            for par, hl in zip(pars[:stop], haslen[:stop])]
    if all(haslen[stop:]):
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

        N=np.diag(toaerrs**2)
        if 'A_rn' in keys:
            plaw = red_noise_powerlaw(A=pars['A_rn'],
                                      alpha=pars['alpha'],
                                      freqs=freqs)
            corr += corr_from_psd(freqs=freqs, psd=plaw, toas=toas)
        M = create_design_matrix(toas, RADEC=True, PROPER=True, PX=True)
        p = sens.Pulsar(toas, toaerrs, phi=pars['phi'][ii],
                        theta=pars['theta'][ii], N=N)
        psrs.append(p)

    return psrs

def sim_SensitivityCurve():
    raise NotImplementedError()
