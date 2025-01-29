# -*- coding: utf-8 -*-
from __future__ import print_function
"""Real data module."""
import numpy as np
from astropy import units as u
import jax.numpy as jnp
import jax.scipy as jsc

import hasasia as hsen

def corr_from_psd_chromatic(freqs, psd, toas, obs_freqs, chr_idx, fref=1400., fast=True):
     """
     Calculates the correlation matrix over a set of TOAs for a given power
     spectral density.

     Parameters
     ----------

     freqs : array
         Array of freqs over which the psd is given.

     psd : array
         Power spectral density to use in calculation of correlation matrix.

     toas : array
         Pulsar times-of-arrival to use in correlation matrix.
     
     obs_freqs : array
         observation frequency of each ToA
         
     chr_idx : float
         Spectral index of the powerlaw amplitude. 2 for DM noise, 4 for Chrom.
         
     fref: float, optional
         reference frequency for amplitude powerlaw. Usually set to 1400 MHz

     fast : bool, optional
         Fast mode uses a matix inner product, while the slower mode uses the
         numpy.trapz function which is slower, but more accurate.

     Returns
     -------

     corr : array
         A 2-dimensional array which represents the correlation matrix for the
         given set of TOAs.
     """

     N_toa = len(toas)
     matrix = np.ones((N_toa,N_toa))

     for i in range(N_toa):
         matrix[i,:] = (fref/obs_freqs[i])**chr_idx
     A_matrix = matrix*matrix.transpose()

     if fast:
         df = np.diff(freqs)
         df = np.append(df,df[-1])
         tm = np.sqrt(psd*df)*np.exp(1j*2*np.pi*freqs*toas[:,np.newaxis])
         integrand = np.matmul(tm, np.conjugate(tm.T))
         return A_matrix*np.real(integrand)
     else: #Makes much larger arrays, but uses np.trapz
         t1, t2 = np.meshgrid(toas, toas, indexing='ij')
         tm = np.abs(t1-t2)
         integrand = psd*np.cos(2*np.pi*freqs*tm[:,:,np.newaxis])#df*
         return A_matrix*np.trapz(integrand, axis=2, x=freqs)#np.sum(integrand,axis=2)#

def get_rednoise_freqs(psr, nmodes, Tspan=None):
    """Frequency components for creating the red noise basis matrix."""

    T = Tspan if Tspan is not None else psr.toas.max() - psr.toas.min()

    f = np.linspace(1 / T, nmodes / T, nmodes)

    Ffreqs = np.zeros(2 * nmodes)
    Ffreqs[::2] = f
    Ffreqs[1::2] = f

    return Ffreqs

def create_fourier_design_matrix(t, nmodes, Tspan=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    :param t: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param Tspan: option to some other Tspan
    :return: F: fourier design matrix
    :return: f: Sampling frequencies
    """

    N = len(t)
    F = np.zeros((N, 2 * nmodes))

    Ffreqs = get_rednoise_freqs(ePsr, nmodes, Tspan=Tspan)

    F[:, ::2] = np.sin(2 * np.pi * t[:, None] * Ffreqs[::2])
    F[:, 1::2] = np.cos(2 * np.pi * t[:, None] * Ffreqs[1::2])

    return F

def get_noise_basis(epsr, toas, nmodes=100):
    """Return a Fourier design matrix for DM noise.

    See the documentation for pl_dm_basis_weight_pair function for details."""

    tbl = toas.table
    t = (tbl["tdbld"].quantity * u.day).to(u.s).value
    fref = 1400 * u.MHz
    D = (fref.value / epsr.freqs) ** 2
    Fmat = create_fourier_design_matrix(t, nmodes)
    return Fmat * D[:, None]

def make_corr(psr):
    N = psr.toaerrs.size
    corr = np.zeros((N,N))
    _, _, fl, _, bi = hsen.quantize_fast(psr.toas,psr.toaerrs,
                                         flags=psr.flags['f'],dt=1)
    keys = [ky for ky in noise.keys() if psr.name in ky]
    backends = np.unique(psr.flags['f'])
    sigma_sqr = np.zeros(N)
    ecorrs = np.zeros_like(fl,dtype=float)
    for be in backends:
        mask = np.where(psr.flags['f']==be)
        key_ef = '{0}_{1}_{2}'.format(psr.name,be,'efac')
        key_eq = '{0}_{1}_log10_{2}'.format(psr.name,be,'t2equad')
        sigma_sqr[mask] = (
                            noise[key_ef]**2 *
                           ((psr.toaerrs[mask])**2 + (10**noise[key_eq])**2)
                          )
        mask_ec = np.where(fl==be)
        key_ec = '{0}_{1}_log10_{2}'.format(psr.name,be,'ecorr')
        ecorrs[mask_ec] = np.ones_like(mask_ec) * (10**noise[key_ec])
    j = [ecorrs[ii]**2*np.ones((len(bucket),len(bucket)))
         for ii, bucket in enumerate(bi)]

    J = sl.block_diag(*j)
    corr = np.diag(sigma_sqr) + J
    return corr

def corr_from_psd_dm(freqs, psd, toas, v_freqs, fast=True):
    """
    Calculates the correlation matrix over a set of TOAs for a given power
    spectral density.

    Parameters
    ----------

    freqs : array
        Array of freqs over which the psd is given.

    psd : array
        Power spectral density to use in calculation of correlation matrix.

    toas : array
        Pulsar times-of-arrival to use in correlation matrix.

    fast : bool, optional
        Fast mode uses a matix inner product, while the slower mode uses the
        numpy.trapz function which is slower, but more accurate.

    Returns
    -------

    corr : array
        A 2-dimensional array which represents the correlation matrix for the
        given set of TOAs.
    """
    if fast:
        df = np.diff(freqs)
        df = np.append(df,df[-1])
        tm = np.sqrt(psd*df)*np.exp(1j*2*np.pi*freqs*toas[:,np.newaxis])*(v_freqs[:,np.newaxis]/1400)**(-2)
        integrand = np.matmul(tm, np.conjugate(tm.T))
        return np.real(integrand)
    else: #Makes much larger arrays, but uses np.trapz
        t1, t2 = np.meshgrid(toas, toas, indexing='ij')
        v1, v2 = np.meshgrid(v_freqs, v_freqs, indexing='ij')
        tm = np.abs(t1-t2)
        integrand = psd*np.cos(2*np.pi*freqs*tm[:,:,np.newaxis])*(v1[:,:,np.newaxis]**-2)*(v2[:,:,np.newaxis]**-2)
        return np.trapz(integrand, axis=2, x=freqs)


def get_red_noise(noise, noise_tag):
    """
    function to find the red noise value associated with psr

    Inputs
    ------
    noise : string
        noise file containing noise values for diff psrs

    noise_tag : string
        can be red noise or dm noise key
        if red noise pass red_noise
        if dm noise pass dm_gp
        if chrom noise pass chrom_gp
    Output
    ------
    psrs[rednoise_key] : red noise value
    """
    key_logA = f'{noise_tag}_log10_A'
    key_gamma = f'{noise_tag}_gamma'
    log10_A_1 = [value for key, value in noise.items() if key_logA in key]
    log10_A_2 = [10**x for x in log10_A_1]
    gamma = [value for key, value in noise.items() if key_gamma in key]
    name = [key for key, value in noise.items() if key_logA in key]
    for i in range(len(name)):
        name[i] = name[i].replace(f"_{noise_tag}_log10_A", "")
    values = [list(t) for t in zip(log10_A_2, gamma)]
    rn_psrs = {}
    for i in name:
        for j in values:
            rn_psrs[i] = j
            values.remove(j)
            break

    return rn_psrs

def hgw_calc(spectra, fyr):
    """
    function to calculate the amplitude of gwb
    for a particular noise curve
    """
    hgw = hsen.Agwb_from_Seff_plaw(spectra.freqs, Tspan=spectra.Tspan,
                                    SNR=5,
                               S_eff=spectra.S_eff)
    plaw_h = hgw *(spectra.freqs/fyr)**(-2/3)
    return hgw, plaw_h


def noise_mod(ePsrs, noise, freqs, dm=False, chrom=False):
    thin = 1
    rn_psrs = get_red_noise(noise, 'red_noise')
    dm_n_psrs = get_red_noise(noise, 'dm_gp')
    chrom_n_psrs = get_red_noise(noise, 'chrom_gp')
    psrs = []

    for ePsr in ePsrs:
        corr = make_corr(ePsr, noise)[::thin,::thin]
        plaw = hsen.red_noise_powerlaw(A=1e-16, gamma=13/3., freqs=freqs)
        corr += hsen.corr_from_psd(freqs=freqs, psd=plaw,
                                toas=ePsr.toas[::thin])
        # if rn:
        if ePsr.name in rn_psrs.keys():
            Amp, gam = rn_psrs[ePsr.name]
            plaw_rn = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)

            corr += hsen.corr_from_psd(freqs=freqs, psd=plaw_rn,
                                toas=ePsr.toas[::thin])
        if dm:
            if ePsr.name in dm_n_psrs.keys():
                Amp, gam = dm_n_psrs[ePsr.name]
                plaw_dm = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)
                corr += hsen.corr_from_psd_dm(freqs=freqs, psd=plaw_dm,
                                    toas=ePsr.toas[::thin], v_freqs=ePsr.freqs)
        if chrom:
            if ePsr.name in chrom_n_psrs.keys():
                Amp, gam = chrom_n_psrs[ePsr.name]
                key_chrom_idx = '{0}_chrom_gp_idx'.format(ePsr.name)
                if key_chrom_idx in noise.keys():
                    plaw_cn = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)
                    corr += hsen.corr_from_psd_chrom(freqs=freqs, psd=plaw_cn,
                                    toas=ePsr.toas[::thin], v_freqs=ePsr.freqs,
                                      index=noise[key_chrom_idx])
                else:
                    plaw_cn = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)
                    corr += hsen.corr_from_psd_chrom(freqs=freqs, psd=plaw_cn,
                                    toas=ePsr.toas[::thin], v_freqs=ePsr.freqs,
                                      index=4.)
        psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                        toaerrs=ePsr.toaerrs[::thin],
                        phi=ePsr.phi,theta=ePsr.theta,
                        N=corr, designmatrix=ePsr.Mmat[::thin,:])
        psr.name = ePsr.name
        psrs.append(psr)
    return psrs