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
import hasasia.utils as hutils

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

    Fmat_red[:, ::2] = jnp.sin(2 * jnp.pi * toas[:, None] * Ffreqs[::2])
    Fmat_red[:, 1::2] = jnp.cos(2 * jnp.pi * toas[:, None] * Ffreqs[1::2])

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

def create_fourier_design_matrix_solar_dm(toas, radio_freqs, planetssb, sunssb, pos_t,
                                       modes=None, nmodes=100,
                                       Tspan=None,):
    """
    Construct DM-Solar Model fourier design matrix.

    :param toas: vector of time series in seconds
    :param planetssb: solar system bayrcenter positions
    :param pos_t: pulsar position as 3-vector
    :param nmodes: number of fourier coefficients to use
    :param freqs: radio frequencies of observations [MHz]
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency

    :return: F: SW DM-variation fourier design matrix
    :return: f: Sampling frequencies
    """

    # get base fourier design matrix and frequencies
    Fmat_red, Ffreqs = create_fourier_design_matrix_achromatic(toas, nmodes=nmodes, Tspan=Tspan)
    # get solar wind dispersion factor
    dt_DM = hutils.solar_wind_geometric_factor(radio_freqs, planetssb, sunssb, pos_t)

    return Fmat_red * dt_DM[:, None], Ffreqs

### flagging convenience functions for determining white noise. modified from Bjorn Larsen's code ####

def get_flags_by_pta(t=None, psr=None, ecorr=False, ppta_band_ecorr=True):
    if not isinstance(t, type(None)):
        flags = t
        B = 'B'
    elif not isinstance(psr, type(None)):
        flags = psr.flags
        if 'B' in list(flags.keys()):
            B = 'B'
        else:
            B = 'b'
    else:
        print('bad input')
        return None
    pta_list = list(np.unique(flags['pta']))
    if ecorr and 'EPTA' in pta_list:
        pta_list.remove('EPTA')
    noiseflags = dict.fromkeys(pta_list)
    for pta in pta_list:
        try:
            groups = np.unique(flags['group'][flags['pta'] == pta])
            try:
                backends = np.unique(flags['f'][flags['pta'] == pta])
            except Exception as e:
                print(f'Exception: {e}')
                print(f'{pta}: using -group')
                noiseflags[pta] = 'group'
            if pta == 'NANOGrav' and 'group' in flags:
                if np.all(backends == groups):
                    noiseflags[pta] = 'group'
                else:
                    noiseflags[pta] = 'f'
            elif pta == 'PPTA' and ecorr and ppta_band_ecorr:
                noiseflags[pta] = B
            else:
                if groups[0] == '' and len(groups) == 1:
                    noiseflags[pta] = 'f'
                else:
                    noiseflags[pta] = 'group'
        except Exception as e:
            print(f"Exception: {e}")
            print(f'{pta}: using -f')
            noiseflags[pta] = 'f'
    return noiseflags

def noise_flags(noiseflags, t=None, psr=None):
    pta_list = list(noiseflags.keys())
    if not isinstance(t, type(None)):
        return np.concatenate([np.unique(t[noiseflags[pta]][t['pta'] == pta])
                               for pta in pta_list])
    elif not isinstance(psr, type(None)):
        f = psr.flags
        return np.concatenate([np.unique(f[noiseflags[pta]][f['pta'] == pta])
                               for pta in pta_list])
    else:
        print('bad input')
        return 0


def get_febes(ePsr):
    """
    based off which flags are in a pulsar's toas,
    returns the unified flags for front end / back end
    """
    ptas_flgs = get_flags_by_pta(psr=ePsr)
    msks = [(ePsr.flags['pta']==pta) for pta in ptas_flgs ]
    flgs = np.concatenate([ePsr.flags[ptas_flgs[pta]][msk] for msk, pta in zip(msks, ptas_flgs)])
    new_flgs=flgs[np.concatenate([ePsr.toas[msk] for msk in msks]).argsort()]
    return new_flgs


def white_noise_corr(psr,
                    noise_dict,
                    equad_convention='tnequad',
                    ecorr_settings='NANOGrav'):
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

    # setup the matrix to be Ntoa x Ntoa
    N = psr.toaerrs.size
    corr = np.zeros((N,N))
    # get the flagging conventions for PTAs
    flags_by_pta = get_flags_by_pta(psr=psr)
    # get the unique frontend/backend combinations
    unique_frontend_backend = noise_flags(flags_by_pta, psr=psr)
    # create a new flag to use
    febes = get_febes(psr)
    # quantize new flag -- i wonder what happens if 2 ptas take measurements at the same time ??
    _, _, flags_quantized, _, bi = hsen.quantize_fast(psr,febes,dt=0.1, flags_only=True)
    sigma_sqr = np.zeros(N)
    ecorrs = np.zeros_like(flags_quantized,dtype=float)
    # FIXME: try to get rid of the the try/ accept
    for unique_febe in unique_frontend_backend:
        mask = np.where(unique_febe==febes)
        try:
            key_ef = '{0}_{1}_{2}'.format(psr.name, unique_febe,'efac')
            efac = noise_dict[key_ef]
        except KeyError:
            efac = 1
            print(f'No efac for {psr.name} {unique_febe}')
        try:
            key_eq = '{0}_{1}_log10_{2}'.format(psr.name, unique_febe,'tnequad')
            equad = 10**noise_dict[key_eq]
        except KeyError:
            print(f'No equad for {psr.name} {unique_febe}')
            equad = 0
        if equad_convention == 'tnequad':
            sigma_sqr[mask] = ( # variance_tn = efac^2 * toaerr^2 + equad^2
                                efac**2 *
                            (psr.toaerrs[mask])**2 + (10**equad)**2
                            )
        elif equad_convention == 't2equad':
            sigma_sqr[mask] = ( # variance_t2 = efac^2 * (toaerr^2 + equad^2)
                                efac**2 *
                            ((psr.toaerrs[mask])**2 + (equad)**2)
                            )
        if ecorr_settings is not None:
            mask_ec = np.where(flags_quantized==unique_febe)
            try:
                key_ec = '{0}_{1}_log10_{2}'.format(psr.name,unique_febe,'ecorr')
                ecorrs[mask_ec] = np.ones_like(mask_ec) * (10**noise_dict[key_ec])
            except KeyError:
                print(f'No ecorr for {psr.name} {unique_febe}')
    if ecorr_settings == 'NANOGrav': # some PTAs don't have ecorr
        j = [ecorrs[ii]**2*np.ones((len(bucket),len(bucket)))
            for ii, bucket in enumerate(bi)]
        J = sl.block_diag(*j)
        corr = jnp.diag(sigma_sqr) + J
    elif ecorr_settings is None:
        corr = jnp.diag(sigma_sqr)
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
         tm = jnp.sqrt(psd*df)*jnp.exp(1j*2*jnp.pi*freqs*toas[:,np.newaxis])
         integrand = jnp.matmul(tm, jnp.conjugate(tm.T))
         return A_matrix*jnp.real(integrand)
     else: #Makes much larger arrays, but uses np.trapz
         t1, t2 = jnp.meshgrid(toas, toas, indexing='ij')
         tm = jnp.abs(t1-t2)
         integrand = psd*jnp.cos(2*jnp.pi*freqs*tm[:,:,jnp.newaxis])#df*
         return A_matrix*jnp.trapz(integrand, axis=2, x=freqs)#np.sum(integrand,axis=2)#


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


def corr_from_psd_solar_wind(toas, radio_freqs, planetssb, sunssb, pos_t, freqs, psd, fast=True):
    """
    Calculates the correlation matrix over a set of TOAs for a given power
    spectral density. Uses a solar wind basis.

    Parameters
    ----------

    toas : array
        Pulsar times-of-arrival to use in correlation matrix.

    radio_freqs : array
         observation frequency of each ToA
    
    planetssb : array

    sunssb : array

    pos_t : array

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

    #  for i in range(N_toa):
    #      matrix[i,:] = (fref/radio_freqs[i])**chromatic_idx
    matrix = hutils.solar_wind_geometric_factor(radio_freqs, planetssb, sunssb, pos_t)[:, np.newaxis]
    A_matrix = matrix*matrix.transpose()

    if fast:
        df = np.diff(freqs)
        df = np.append(df,df[-1])
        tm = jnp.sqrt(psd*df)*jnp.exp(1j*2*jnp.pi*freqs*toas[:,jnp.newaxis])
        integrand = jnp.matmul(tm, jnp.conjugate(tm.T))
        return A_matrix*jnp.real(integrand)
    else: #Makes much larger arrays, but uses np.trapz
        t1, t2 = jnp.meshgrid(toas, toas, indexing='ij')
        tm = jnp.abs(t1-t2)
        integrand = psd*jnp.cos(2*jnp.pi*freqs*tm[:,:,jnp.newaxis])#df*
        return A_matrix*jnp.trapz(integrand, axis=2, x=freqs)#np.sum(integrand,axis=2)#


def get_noise_values(noise_dict, noise_tag):
    """
    function to find the red noise value associated with psr

    Inputs
    ------
    noise_dict : string
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
    log10_A_1 = [value for key, value in noise_dict.items() if key_logA in key]
    log10_A_2 = [10**x for x in log10_A_1]
    gamma = [value for key, value in noise_dict.items() if key_gamma in key]
    name = [key for key, value in noise_dict.items() if key_logA in key]
    for i in range(len(name)):
        name[i] = name[i].replace(f"_{noise_tag}_log10_A", "")
    values = [list(t) for t in zip(log10_A_2, gamma)]
    noise_vals = {}
    for i in name:
        for j in values:
            noise_vals[i] = j
            values.remove(j)
            break
    return noise_vals

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


def make_corr(ePsr, noise_dict, freqs,
              include_achrom_rn_corr=False,
              include_dmgp_corr=False,
              include_chromgp_corr=False,
              include_swgp_corr=False,
              thin=1, A_gwb=1e-16, gamma_gwb=13/3.,
              equad_convention='tnequad', ecorr_settings='NANOGrav'):
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

    include_dmgp_corr : bool, optional
        Option to include dm noise in the correlation matrix.

    include_chromgp_corr : bool, optional
        Option to include chromatic noise in the correlation matrix.

    include_swgp_corr : bool, optional
        Option to include swgp noise in the correlation matrix.

    thin : int, optional
        Option to thin the toas for faster calculations.
        Calculations run faster, but overall sensitivity is reduced as the power in the white nosie is higher.
    
    A_gwb : float, optional
        Amplitude of the GWB

    gamma_gwb : float, optional
        Spectral index of the GWB

    """
    # thin the toas at the onset for maximal efficiency
    ePsr.toas = ePsr.toas[::thin]
    ePsr.toaerrs = ePsr.toaerrs[::thin]
    ePsr.Mmat = ePsr.Mmat[::thin, :]

    rn_psrs = get_noise_values(noise_dict, 'red_noise')
    dm_n_psrs = get_noise_values(noise_dict, 'dm_gp')
    chrom_n_psrs = get_noise_values(noise_dict, 'chrom_gp')
    swgp_psrs = get_noise_values(noise_dict, 'sw_gp')

    corr = white_noise_corr(ePsr,
                            noise_dict,
                            equad_convention=equad_convention,
                            ecorr_settings=ecorr_settings)
    plaw = hsen.red_noise_powerlaw(A_gwb=A_gwb, gamma_gwb=gamma_gwb, freqs=freqs)
    corr += hsen.corr_from_psd(freqs=freqs, psd=plaw,
                            toas=ePsr.toas)

    if include_achrom_rn_corr:
        if ePsr.name in rn_psrs.keys():
            Amp, gam = rn_psrs[ePsr.name]
            plaw_rn = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)
            corr += hsen.corr_from_psd(freqs=freqs, psd=plaw_rn,
                            toas=ePsr.toas)
    if include_dmgp_corr:
        if ePsr.name in dm_n_psrs.keys():
            Amp, gam = dm_n_psrs[ePsr.name]
            plaw_dm = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)
            corr += hsen.corr_from_psd_dm(ePsr.toas, ePsr.freqs, freqs, plaw_dm)
    if include_chromgp_corr:
        if ePsr.name in chrom_n_psrs.keys():
            Amp, gam = chrom_n_psrs[ePsr.name]
            key_chrom_idx = '{0}_chrom_gp_idx'.format(ePsr.name)
            if key_chrom_idx in noise_dict.keys():
                plaw_chrom = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)
                corr += corr_from_psd_dm(
                    ePsr.toas,
                    ePsr.freqs,
                    freqs,
                    plaw_chrom,
                    chromatic_index=noise_dict[key_chrom_idx]
                    )
            else:
                plaw_chrom = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)
                corr += corr_from_psd_chromatic(
                    ePsr.toas,
                    ePsr.freqs,
                    freqs,
                    psd=plaw_chrom,
                    index=4.
                    )
    if include_swgp_corr:
        if ePsr.name in swgp_psrs.keys():
            Amp, gam = swgp_psrs[ePsr.name]
            key_chrom_idx = '{0}_sw_gp_idx'.format(ePsr.name)
            plaw_chrom = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)
            corr += corr_from_psd_solar_wind(
                ePsr.toas,
                ePsr.freqs,
                freqs,
                plaw_chrom,
                )
    return corr


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
    freqs = jnp.logspace(jnp.log10(1/(5*Tspan)),jnp.log10(4e-7),600)

    psr = noise_corr(ePsrs, noise, freqs, dm, chrom)
    specs = [hsen.Spectrum(p, freqs=freqs) for p in psr]
    pta = hsen.GWBSensitivityCurve(specs)
    hgw, plaw_h= hgw_calc(pta, fyr)
    print('hgw = ', hgw)
    return specs, pta, hgw