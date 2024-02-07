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
            A_rn=None, alpha=None, freqs=None, uneven=False,
            A_gwb=None, gamma_gwb = 13/3, fast=True, psr_names = None,
            kwastro={'RADEC':True, 'PROPER':True, 'PX':True}):
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

    uneven : bool, optional
        Option to have the toas be unevenly sampled.

    fast : bool, optional
        Option to use the faster, less accurate PSD-to-correlation matrix
        calculation.

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
    Timespan = np.amax(pars['timespan'])
    if psr_names is None:
        psr_names = [None for _ in range(Npsrs)]
    for ii in range(Npsrs):
        tspan = pars['timespan'][ii]
        Ntoas = int(np.floor(tspan*pars['cad'][ii]))
        delay = Timespan - tspan
        toas = np.linspace(delay, Timespan, Ntoas)*yr_sec
        if uneven:
            dt = tspan / Ntoas / 4 * yr_sec
            toas += np.random.uniform(-dt, dt, size=toas.size)

        if err_dim == 2:
            toaerrs = pars['sigma'][ii,:]
        else:
            toaerrs = pars['sigma'][ii]*np.ones(Ntoas)

        N = np.diag(toaerrs**2)
        if 'A_rn' in keys:
            plaw = red_noise_powerlaw(A=pars['A_rn'][ii],
                                      alpha=pars['alpha'][ii],
                                      freqs=freqs)
            N += corr_from_psd(freqs=freqs, psd=plaw, toas=toas, fast=fast)

        if A_gwb is not None:
            gwb = red_noise_powerlaw(A=A_gwb,
                                     alpha=gamma_gwb,
                                     freqs=freqs)
            N += corr_from_psd(freqs=freqs, psd=gwb, toas=toas, fast=fast)

        M = create_design_matrix(toas, **kwastro)

        p = Pulsar(toas, toaerrs, phi=pars['phi'][ii], name=psr_names[ii],
                   theta=pars['theta'][ii], designmatrix=M, N=N)
        psrs.append(p)

    return psrs
