# -*- coding: utf-8 -*-
from __future__ import print_function
"""Real data module."""
import numpy as np
import scipy.linalg as sl
from astropy import units as u
import jax.numpy as jnp
import jax.scipy as jsc
import glob
import hasasia.sensitivity as hsen

fyr = 1/(365.25*24*3600)


## could add a thinning here to make it faster.


def get_fourier_freqs(nmodes, Tspan=None):
    """Frequency components for creating the fourier basis design matrices."""

    f = np.linspace(1 / Tspan, nmodes / Tspan, nmodes)

    Ffreqs = np.zeros(2 * nmodes)
    Ffreqs[::2] = f
    Ffreqs[1::2] = f

    return Ffreqs


def get_psrname(file,name_sep='_'):
    return (file.split('/')[-1].split(name_sep)[0]).split(".")[0]


def create_fourier_design_matrix_achromatic(toas, nmodes, Tspan=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    :param toas: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param Tspan: option to some other Tspan
    :return: F: fourier design matrix
    :return: f: Sampling frequencies
    """

    N = len(toas)
    Fmat_red = np.zeros((N, 2 * nmodes))

    Ffreqs = get_fourier_freqs(nmodes=nmodes, Tspan=Tspan)

    Fmat_red[:, ::2] = np.sin(2 * np.pi * toas[:, None] * Ffreqs[::2])
    Fmat_red[:, 1::2] = np.cos(2 * np.pi * toas[:, None] * Ffreqs[1::2])

    return Fmat_red

def create_fourier_design_matrix_chromatic(toas, radio_freqs, nmodes, Tspan=None, chrom_idx=4., fref=1400.):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    :param toas: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param Tspan: option to some other Tspan
    :return: Fmat: chromatic Fourier design matrix
    :return: Ffreqs: Sampling frequencies
    """

    N = len(toas)
    Fmat = np.zeros((N, 2 * nmodes))

    Ffreqs = get_fourier_freqs(nmodes=nmodes, Tspan=Tspan)
    D = (fref / radio_freqs) ** chrom_idx

    create_fourier_design_matrix_achromatic(toas=toas, nmodes=nmodes, Tspan=Tspan)

    return Fmat * D[:, None], Ffreqs

def create_fourier_design_matrix_dm(toas, nmodes, Tspan=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    :param toas: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param Tspan: option to some other Tspan
    :return: F: fourier design matrix
    :return: f: Sampling frequencies
    """
    Fmat_dm = create_fourier_design_matrix_chromatic(toas=toas, nmodes=nmodes, Tspan=Tspan, chrom_idx=2.0)

    return Fmat_dm

def white_noise_corr(psr,
                    noise_dict,
                    equad_convention='tnequad',
                    include_ecorr=True):
    """
    Function to make white noise correlation matrix.
    Formerly, `make_corr` in the real data tutorial.
    Makes the white noise portion of the correlation matrix.
    Stochastic signals can subsequently be added to this matrix.

    Parameters
    ----------
    psr : enterprise.pulsar.Pulsar object
        The pulsar object for which to create the correlation matrix.
    noise_dict: dictionary
        Dictionary containing noise parameters.
    equad_convention : str, optional
        Convention for equad parameter. Can be 'tnequad' or 't2equad'.
    include_ecorr : bool, optional
        Option to include ecorr in the white noise correlation matrix.
        Should be turned on if the ecorr values are in the noise_dict.

    Returns
    -------
    corr : array (Ntoa x Ntoa)
        The white noise correlation matrix.
    """
    ### formally make_corr
    N = psr.toaerrs.size
    corr = np.zeros((N,N))
    ### need to finagle some flags here
    _, _, fl, _, bi = hsen.quantize_fast(psr.toas,psr.toaerrs,
                                         flags=psr.flags['f'],dt=1)
    keys = [ky for ky in noise_dict.keys() if psr.name in ky]
    backends = np.unique(psr.flags['f'])
    sigma_sqr = np.zeros(N)
    ecorrs = np.zeros_like(fl,dtype=float)
    for be in backends:
        #### FIXME: check the flags
        mask = np.where(psr.flags['f']==be)
        key_ef = '{0}_{1}_{2}'.format(psr.name,be,'efac')
        key_eq = '{0}_{1}_log10_{2}'.format(psr.name,be,equad_convention)
        if equad_convention == 'tnequad':
            sigma_sqr[mask] = ( # variance_tn = efac^2 * toaerr^2 + equad^2
                                noise_dict[key_ef]**2 *
                            (psr.toaerrs[mask])**2 + (10**noise_dict[key_eq])**2
                            )
        elif equad_convention == 't2equad':
            sigma_sqr[mask] = ( # variance_t2 = efac^2 * (toaerr^2 + equad^2)
                                noise_dict[key_ef]**2 *
                            ((psr.toaerrs[mask])**2 + (noise_dict[key_eq])**2)
                            )
        if include_ecorr:
            mask_ec = np.where(fl==be)
            key_ec = '{0}_{1}_log10_{2}'.format(psr.name,be,'ecorr')
            ecorrs[mask_ec] = np.ones_like(mask_ec) * (10**noise_dict[key_ec])
    if include_ecorr: # some PTAs don't have ecorr
        j = [ecorrs[ii]**2*np.ones((len(bucket),len(bucket)))
            for ii, bucket in enumerate(bi)]
        J = sl.block_diag(*j)
        corr = np.diag(sigma_sqr) + J
    else:
        corr = np.diag(sigma_sqr)
    return corr

def corr_from_psd_chromatic(toas, radio_freqs, freqs, psd, chromatic_idx, fref=1400., fast=True):
     """
     Calculates the correlation matrix over a set of TOAs for a given power
     spectral density.

     Parameters
     ----------
     
     toas : array
         Pulsar times-of-arrival to use in correlation matrix.

     radio_freqs : array
         observation frequency of each ToA

     freqs : array
         Array of freqs over which the psd is given.

     psd : array
         Power spectral density to use in calculation of correlation matrix.

    chromatic_idx : float
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
         matrix[i,:] = (fref/radio_freqs[i])**chromatic_idx
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


def corr_from_psd_dm(toas, radio_freqs, freqs, psd, fref=1400., fast=True):
     """
     Calculates the correlation matrix over a set of TOAs for a given power
     spectral density.
     Wrapper around `corr_from_psd_chromatic` with `chromatic_index=2`.

     Parameters
     ----------

    toas : array
         Pulsar times-of-arrival to use in correlation matrix.

     radio_freqs : array
         observation frequency of each ToA

     freqs : array
         Array of freqs over which the psd is given.

     psd : array
         Power spectral density to use in calculation of correlation matrix.

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
     return corr_from_psd_chromatic(toas=toas, radio_freqs=radio_freqs, freqs=freqs, psd=psd, chr_idx=2.0, fref=fref, fast=fast)


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


def make_corr(ePsr, noise, freqs, dm=False, chrom=False, thin=1, A_gwb=1e-16, gamma_gwb=13/3.):
    """
    function to make the correlation matrix for a pulsar
    Parameters
    ----------
    ePsr : enterprise.pulsar.Pulsar object
        The pulsar object for which to create the correlation matrix.

    noise : dictionary
        Dictionary containing noise parameters.

    freqs : array
        Array of freqs over which the psd is given.

    dm : bool, optional
        Option to include dm noise in the correlation matrix.

    chrom : bool, optional
        Option to include chromatic noise in the correlation matrix.

    thin : int, optional
        Option to thin the toas for faster calculations.
        Calculations run faster, but overall sensitivity is reduced as the power in the white nosie is higher.
    
    A_gwb : float, optional
        Amplitude of the GWB

    gamma_gwb : float, optional
        Spectral index of the GWB

    """
    ePsr.toas = ePsr.toas[::thin]
    ePsr.toaerrs = ePsr.toaerrs[::thin]
    ePsr.Mmat = ePsr.Mmat[::thin,:]
    rn_psrs = get_red_noise(noise, 'red_noise')
    dm_n_psrs = get_red_noise(noise, 'dm_gp')
    chrom_n_psrs = get_red_noise(noise, 'chrom_gp')

    corr = white_noise_corr(ePsr, noise)[::thin,::thin]
    plaw = hsen.red_noise_powerlaw(A_gwb=1e-16, gamma_gwb=13/3., freqs=freqs)
    corr += hsen.corr_from_psd(freqs=freqs, psd=plaw,
                            toas=ePsr.toas)
    # if rn:
    if ePsr.name in rn_psrs.keys():
        Amp, gam = rn_psrs[ePsr.name]
        plaw_rn = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)

        corr += hsen.corr_from_psd(freqs=freqs, psd=plaw_rn,
                            toas=ePsr.toas)
    if dm:
        if ePsr.name in dm_n_psrs.keys():
            Amp, gam = dm_n_psrs[ePsr.name]
            plaw_dm = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)
            corr += hsen.corr_from_psd_dm(freqs=freqs, psd=plaw_dm,
                                toas=ePsr.toas, v_freqs=ePsr.freqs)
    if chrom:
        if ePsr.name in chrom_n_psrs.keys():
            Amp, gam = chrom_n_psrs[ePsr.name]
            key_chrom_idx = '{0}_chrom_gp_idx'.format(ePsr.name)
            if key_chrom_idx in noise.keys():
                plaw_cn = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)
                corr += hsen.corr_from_psd_chrom(freqs=freqs, psd=plaw_cn,
                                toas=ePsr.toas, v_freqs=ePsr.freqs,
                                    index=noise[key_chrom_idx])
            else:
                plaw_cn = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)
                corr += hsen.corr_from_psd_chrom(freqs=freqs, psd=plaw_cn,
                                toas=ePsr.toas, v_freqs=ePsr.freqs,
                                    index=4.)
    psr = hsen.Pulsar(toas=ePsr.toas,
                      toaerrs=ePsr.toaerrs,
                      phi=ePsr.phi,theta=ePsr.theta,
                      N=corr, designmatrix=ePsr.Mmat)
    psr.name = ePsr.name
    return psr


def calc_pta_gw(parpath, timpath, psrlist, noise, dm=False, chrom=False):
    """
    function to calculate the hgw for mpta
    Input:
    pars, tims, 

    Output:
    hgw
    
    """
    pars = sorted(glob.glob(parpath+'*.par'))
    tims = sorted(glob.glob(timpath+'*.tim'))

    pars = [f for f in pars if get_psrname(f) in psrlist]
    tims = [f for f in tims if get_psrname(f) in psrlist]


    ePsrs = create_ePsrs(pars, tims)

    Tspan = hsen.get_Tspan(ePsrs)
    freqs = np.logspace(np.log10(1/(5*Tspan)),np.log10(4e-7),600)

    psr = noise_corr(ePsrs, noise, freqs, dm, chrom)
    specs = [hsen.Spectrum(p, freqs=freqs) for p in psr]
    pta = hsen.GWBSensitivityCurve(specs)
    hgw, plaw_h= hgw_calc(pta, fyr)
    print('hgw = ', hgw)
    return specs, pta, hgw