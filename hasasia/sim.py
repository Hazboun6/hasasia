# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np
from .sensitivity import Pulsar, red_noise_powerlaw, corr_from_psd
from .utils import create_design_matrix
__all__ = ['sim_pta',
           ]

day_sec = 24*3600
yr_sec = 365.25*24*3600

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

    Returns
    -------
    psrs : list
        List of `hasasia.Pulsar()` objects.

    """
    #Automatically deal with single floats and arrays.
    if A_rn is None and alpha is None:
        pars = [timespan, cad, sigma, phi, theta]
        keys = ['timespan', 'cad', 'sigma', 'phi', 'theta']
        stop = 3
    # elif None in [A_rn, alpha, freqs]:
    #     err_msg = 'A_rn, alpha and freqs must all be specified '
    #     err_msg += 'in order to build C_rn.'
    #     raise ValueError(err_msg)
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

        N = np.diag(toaerrs**2)
        if 'A_rn' in keys:
            plaw = red_noise_powerlaw(A=pars['A_rn'][ii],
                                      alpha=pars['alpha'][ii],
                                      freqs=freqs)
            N += corr_from_psd(freqs=freqs, psd=plaw, toas=toas)
        M = create_design_matrix(toas, RADEC=True, PROPER=True, PX=True)
        p = Pulsar(toas, toaerrs, phi=pars['phi'][ii],
                   theta=pars['theta'][ii], N=N)
        psrs.append(p)

    return psrs

def sim_SensitivityCurve():
    raise NotImplementedError()
