# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np
import itertools as it
import scipy.stats as sps
import scipy.linalg as sl
import os, pickle, jax, h5py
from astropy import units as u
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsc
from functools import cached_property, partial
import hasasia
from .utils import create_design_matrix, theta_phi_to_SkyCoord, skycoord_to_Jname
current_path = os.path.abspath(hasasia.__path__[0])
sc_dir = os.path.join(current_path,'sensitivity_curves/')

__all__ =['GWBSensitivityCurve',
          'DeterSensitivityCurve',
          'Pulsar',
          'Spectrum',
          'Spectrum_RRF',
          'R_matrix',
          'G_matrix',
          'get_Tf',
          'get_NcalInv',
          'get_NcalInv_RRF',
          'resid_response',
          'HellingsDownsCoeff',
          'get_Tspan',
          'get_TspanIJ',
          'corr_from_psd',
          'quantize_fast',
          'red_noise_powerlaw',
          'Agwb_from_Seff_plaw',
          'PI_hc',
          'nanograv_11yr_stoch',
          'nanograv_11yr_deter',
          ]

## Some constants
yr_sec = 365.25*24*3600
fyr = 1/yr_sec

def R_matrix(designmatrix, N):
    """
    Create R matrix as defined in Ellis et al (2013)
    and Demorest et al (2012)

    Parameters
    ----------

    designmatrix : array
        Design matrix of timing model.

    N : array
        TOA uncertainties [s]

    Returns
    -------
    R matrix

    """
    M = designmatrix
    n,m = M.shape
    L = np.linalg.cholesky(N)
    Linv = np.linalg.inv(L)
    U,s,_ = np.linalg.svd(jnp.matmul(Linv,M), full_matrices=True)
    Id = np.eye(M.shape[0])
    S = np.zeros_like(M)
    S[:m,:m] = np.diag(s)
    inner = np.linalg.inv(jnp.matmul(S.T,S))
    outer = jnp.matmul(S,jnp.matmul(inner,S.T))

    return Id - jnp.matmul(L,jnp.matmul(jnp.matmul(U,outer),jnp.matmul(U.T,Linv)))

def G_matrix(designmatrix):
    """
    Create G matrix as defined in van Haasteren 2013

    Parameters
    ----------

    designmatrix : array
        Design matrix for a pulsar timing model.

    Returns
    -------
    G matrix

    """
    M = designmatrix
    n , m = M.shape
    U, _ , _ = np.linalg.svd(M, full_matrices=True)

    return U[:,m:]

def get_Tf(designmatrix, toas, N=None, nf=200, fmin=None, fmax=2e-7,
           freqs=None, exact_astro_freqs = False,
           from_G=True, twofreqs=False, Gmatrix=None):
    """
         the transmission function for a given pulsar design matrix, TOAs
    and TOA errors.

    Parameters
    ----------

    designmatrix : array
        Design matrix for a pulsar timing model, N_TOA x N_param.

    toas : array
        Times-of-arrival for pulsar, N_TOA long.

    N : array
        Covariance matrix for pulsar time-of-arrivals, N_TOA x N_TOA. Often just
        a diagonal matrix of inverse TOA errors squared.

    nf : int, optional
        Number of frequencies at which to calculate transmission function.

    fmin : float, optional
        Minimum frequency at which to calculate transmission function.

    fmax : float, optional
        Maximum frequency at which to calculate transmission function.

    exact_astro_freqs : bool, optional
        Whether to use exact 1/year and 2/year frequency values in calculation.

    from_G : bool, optional
        Whether to use G matrix for transmission function calculate. If False
        R-matrix is used.

    twofreqs : bool, optional
        Whether to calculate a two frequency transmission function.

    Gmatrix : ndarray, optional
        Provide already calculated G-matrix. This can speed up calculations
        since the singular value decomposition can take time for large matrices.
    """
    if not from_G and N is None:
        err_msg = 'Covariance Matrix must be provided if constructing'
        err_msg += ' from R-matrix.'
        raise ValueError(err_msg)

    M = designmatrix
    N_TOA = M.shape[0]
    ## Prep Correlation
    t1, t2 = np.meshgrid(toas, toas)
    tm = np.abs(t1-t2)

    # make filter
    T = toas.max()-toas.min()
    f0 = 1 / T
    if freqs is None:
        if fmin is None:
            fmin = f0/5
        ff = np.logspace(np.log10(fmin), np.log10(fmax), nf,dtype='float64')
        if exact_astro_freqs:
            ff = np.sort(np.append(ff,[fyr,2*fyr]))
            nf +=2
    else:
        nf = len(freqs)
        ff = freqs

    Tmat = np.zeros(nf, dtype='float64')
    if from_G:
        if Gmatrix is None:
            G = G_matrix(M)
        else:
            G = Gmatrix
        m = G.shape[1]
        Gtilde = jnp.zeros((ff.size, G.shape[1]), dtype=jnp.complex64)
        Gtilde = np.dot(np.exp(1j*2*np.pi*ff[:,np.newaxis]*toas),G).astype(jnp.complex64)
        Tmat = jnp.matmul(jnp.conjugate(Gtilde),Gtilde.T)/N_TOA
        if twofreqs:
            Tmat = np.real(Tmat)
        else:
            Tmat = np.real(np.diag(Tmat))
    else:
        R = R_matrix(M, N)
        for ct, f in enumerate(ff):
            Tmat[ct] = np.real(np.sum(np.exp(1j*2*np.pi*f*tm)*R)/N_TOA)

    return np.real(Tmat), ff, T

def get_NcalInv(psr, nf=200, fmin=None, fmax=2e-7, freqs=None,
                exact_yr_freqs = False, full_matrix=False,
                return_Gtilde_Ncal=False, tm_fit=True, Gmatrix=None):
    r"""
    Calculate the inverse-noise-wieghted transmission function for a given
    pulsar. This calculates
    :math:`\mathcal{N}^{-1}(f,f') , \; \mathcal{N}^{-1}(f)`
    in `[1]`_, see Equations (19-20).

    .. _[1]: https://arxiv.org/abs/1907.04341

    Parameters
    ----------

    psr : array
        Pulsar object.

    nf : int, optional
        Number of frequencies at which to calculate transmission function.

    fmin : float, optional
        Minimum frequency at which to calculate transmission function.

    fmax : float, optional
        Maximum frequency at which to calculate transmission function.

    exact_yr_freqs : bool, optional
        Whether to use exact 1/year and 2/year frequency values in calculation.

    full_matrix : bool, optional
        Whether to return the full, two frequency NcalInv.

    return_Gtilde_Ncal : bool, optional
        Whether to return Gtilde and Ncal. Gtilde is the Fourier transform of
        the G-matrix.

    tm_fit : bool, optional
        Whether to include the timing model fit in the calculation.

    Gmatrix : ndarray, optional
        Provide already calculated G-matrix. This can speed up calculations
        since the singular value decomposition can take time for large matrices.

    Returns
    -------

    inverse-noise-weighted transmission function

    """
    toas = psr.toas
    # make filter
    T = toas.max()-toas.min()
    f0 = 1 / T
    if freqs is None:
        if fmin is None:
            fmin = f0/5
        ff = np.logspace(np.log10(fmin), np.log10(fmax), nf,dtype='float128')
        if exact_yr_freqs:
            ff = np.sort(np.append(ff,[fyr,2*fyr]))
            nf +=2
    else:
        nf = len(freqs)
        ff = freqs

    if tm_fit:
        if Gmatrix is None:
            G = G_matrix(psr.designmatrix)
        else:
            G = Gmatrix
    else:
        G = np.eye(toas.size)

    if hasattr(psr,'N'):
        L = jsc.linalg.cholesky(psr.N)            
        A = jnp.matmul(L,G)
        del L
        N_TMM = jnp.matmul(A.T,A)
        del A
        NInv_TMM = jnp.linalg.inv(N_TMM)
    else:
        NInv_TMM = psr.K_inv


    Gtilde = np.zeros((ff.size,G.shape[1]),dtype='complex128')
    #N_freqs x N_TOA-N_par

    # Note we do not include factors of NTOA or Timespan as they cancel
    # with the definition of Ncal
    Gtilde = np.dot(np.exp(1j*2*np.pi*ff[:,np.newaxis]*toas),G)
    # N_freq x N_TOA-N_par

   
    TfN = jnp.matmul(np.conjugate(Gtilde),jnp.matmul(NInv_TMM,Gtilde.T)) / 2
    if return_Gtilde_Ncal:
        return np.real(TfN), Gtilde, jnp.linalg.inv(NInv_TMM)
    elif full_matrix:
        return np.real(TfN)
    else:
        return np.real(np.diag(TfN)) / get_Tspan([psr])
    

def get_KIJ_Inv(spectra:list, rn_psrs:dict):
    r"""Timing model marginalized inverse covariance matrix.

    .. math::
    [K^{-1}]_{IJ} \equiv G_{I} [(G^{T} C G)^{-1}]_{IJ} G_{J}^{T}
    - [(G^{T} C G)]_{IJ} = G_{I}^{T} (C^{h}_{IJ} + \delta_{IJ}N_{I})G_{J}
    """
    Npsrs = len(spectra)
    psr_idx = np.arange(Npsrs)
    pairs = list(it.combinations(psr_idx,2))

    gwb = red_noise_powerlaw(A=spectra[0].A_gwb, gamma=spectra[0].gamma_gwb, freqs=spectra[0].freqs)

    #TMM block noise covariance
    K_IJ_block = np.empty((Npsrs, Npsrs), dtype=object)
    G_dims = []
    II = 0
    #for-loop to go over every unique pair and diagonal element
    for I, J in pairs:
        Chi_IJ = HD([spectra[I].phi, spectra[J].phi], [spectra[I].theta, spectra[J].theta])
        #Signal covariance matrix for single pulsar pair, strictly for the off-diagonal terms
        Ch_IJ = Chi_IJ * corr_from_psdIJ(freqs=spectra[0].freqs, psd=gwb, toasI=spectra[I].toas, toasJ=spectra[J].toas)

        K_off_diag = jnp.matmul(spectra[I].G.T, jnp.matmul(Ch_IJ, spectra[J].G))    
        K_IJ_block[I,J] = K_off_diag
        K_IJ_block[J,I] = K_off_diag.T

        #diagonal entries
        if II < Npsrs:
            plaw = gwb
            C = spectra[II].N
            if spectra[II].name in rn_psrs.keys():
                Amp, gam = rn_psrs[spectra[II].name]
                plaw += red_noise_powerlaw(A=Amp, gamma=gam, freqs=spectra[0].freqs)

            C += corr_from_psd(freqs=spectra[0].freqs, psd=plaw, toas=spectra[II].toas)

            K_IJ_block[II,II] = jnp.matmul(spectra[II].G.T, jnp.matmul(C, spectra[II].G))
            G_dims.append(spectra[II].G.shape[1])
            II += 1

    
    #convert TMM block noise covariance to TMM noise covariance
    K_IJ = np.zeros((sum(G_dims), sum(G_dims)), dtype=np.float64)
    row_start = 0
    for i in range(Npsrs):  
        col_start = 0
        for j in range(Npsrs): 
            K_IJ[row_start:row_start + G_dims[i], col_start:col_start + G_dims[j]] = K_IJ_block[i,j]
            K_IJ_block[i,j] = 0
            col_start += G_dims[j]  
        row_start += G_dims[i]  

    del K_IJ_block

    #computes TMM inverse noise covariance
    KIJ_Inv = np.linalg.inv(K_IJ)
    return KIJ_Inv


def get_KII_Inv(spectra:list, rn_psrs:dict):
    r"""Timing model marginalized inverse covariance matrix with diagonal approximation.

    .. math::
    [K^{-1}]_{II} \equiv G_{I} [(G^{T} C G)^{-1}]_{II} G_{I}^{T}
    """
    Npsrs = len(spectra)
    psr_idx = np.arange(Npsrs)
    G_dims = []

    for i in range(Npsrs):
        G_dims.append(spectra[i].G.shape[1])


    KIJ_Inv = np.zeros((sum(G_dims), sum(G_dims)), dtype=np.float64)
    row_start=0
    for i in range(Npsrs):  
        if not hasattr(spectra[i], 'K_Inv'):
            K = jnp.matmul(spectra[i].G.T, jnp.matmul(spectra[i].N, spectra[i].G))
            K_Inv = jnp.linalg.inv(K)
            KIJ_Inv[row_start:row_start + G_dims[i], row_start:row_start + G_dims[i]] = K_Inv

        else:                     
            KIJ_Inv[row_start:row_start + G_dims[i], row_start:row_start + G_dims[i]] = spectra[i].K_Inv
        
        row_start += G_dims[i]  

    return KIJ_Inv


def get_NcalInv_IJ(spectra:list, rn_psrs:dict):
    r"""Timing model marginalized inverse noise-weighted transmission function
    """
    Npsrs = len(spectra)
    psr_idx = np.arange(Npsrs)
    pairs = list(it.combinations(psr_idx,2))
    freqs = spectra[0].freqs
    Tspan = get_Tspan(spectra)

    G_dims = []
    for I in range(Npsrs):
        G_dims.append(spectra[I].G.shape[1])

    KIJ_Inv_block = np.empty((Npsrs, Npsrs), dtype=object)
    KIJ_Inv = get_KIJ_Inv(spectra, rn_psrs)

    row_start = 0
    for i in range(Npsrs):  
        col_start = 0
        for j in range(Npsrs):
            KIJ_Inv_block[i,j] = KIJ_Inv[row_start:row_start + G_dims[i], col_start:col_start + G_dims[j]]
            col_start += G_dims[j]  
        row_start += G_dims[i]
    del KIJ_Inv

    NcalInvIJ = np.zeros((Npsrs, Npsrs, freqs.size))

    II = 0
    for I,J in pairs:
        GtildeI = np.zeros((freqs.size, spectra[I].G.shape[1]),dtype='complex128')
        GtildeI = np.dot(np.exp(1j*2*np.pi*freqs[:,np.newaxis]*spectra[I].toas),spectra[I].G)

        GtildeJ = np.zeros((freqs.size, spectra[J].G.shape[1]),dtype='complex128')
        GtildeJ = np.dot(np.exp(1j*2*np.pi*freqs[:,np.newaxis]*spectra[J].toas),spectra[J].G)

        
        NcalInvIJ_entry = jnp.matmul(jnp.conjugate(GtildeI),jnp.matmul(KIJ_Inv_block[I,J],GtildeJ.T)) / 2

        KIJ_Inv_block[I,J] = 0

        NcalInvJI_entry = jnp.conjugate(NcalInvIJ_entry).T
 
        NcalInvIJ[I,J,:] = np.real(np.diag(NcalInvIJ_entry)) / Tspan
        del NcalInvIJ_entry 
        NcalInvIJ[J,I,:] = np.real(np.diag(NcalInvJI_entry)) / Tspan
        del NcalInvJI_entry 

        if II < Npsrs:    
            GtildeII = np.zeros((freqs.size, spectra[II].G.shape[1]),dtype='complex128')
            GtildeII = np.dot(np.exp(1j*2*np.pi*freqs[:,np.newaxis]*spectra[II].toas),spectra[II].G)

            NcalInvII_entry = jnp.matmul(jnp.conjugate(GtildeII),jnp.matmul(KIJ_Inv_block[II,II],GtildeII.T)) / 2

            KIJ_Inv_block[II,II] = 0

            NcalInvIJ[II,II,:] = np.real(np.diag(NcalInvII_entry)) / Tspan
            del NcalInvII_entry

            II+=1

    del KIJ_Inv_block

    return NcalInvIJ


def get_FIM_fastPTA_Approx(spectra:list, gamma_gwb:float, A_gwb:float, rn_psrs:dict):
    r"""
    Fisher information matrix from Babak et. al . See Equation (26) in `[2]`_.

    .. math::
        \mathcal{F}_{\alpha, \beta} = \sum_{i}\frac{1}{2} \mathrm{Tr} \left[ \tilde{C}^{-1}(f_i) \partial_{\alpha}\tilde{C}^{h}(f_i)  \tilde{C}(f_i)^{-1} \partial_{\beta}\tilde{C}^{h}(f_i) \right]

    .. _[2]: https://arxiv.org/abs/2404.02864
    """

    Npsrs = len(spectra)
    psr_idx = np.arange(Npsrs)
    pairs = list(it.combinations(psr_idx,2))
    freqs = spectra[0].freqs
    Tspan = get_Tspan(spectra)

    RIJ = np.zeros((Npsrs, Npsrs, freqs.size), dtype=np.float64)

    II = 0
    for I, J in pairs:
        #spectra pulsars
        
        TIJ = min(get_Tspan([spectra[I]]), get_Tspan([spectra[J]]))
        Chi_IJ = HD([spectra[I].phi, spectra[J].phi], [spectra[I].theta, spectra[J].theta])

        #response tensors
        RIJ[I,J,:] = Chi_IJ* np.sqrt(spectra[I].Tf*spectra[J].Tf*TIJ/Tspan)
        RIJ[J,I,:] = RIJ[I,J,:]

        #conditional for diagonal elements
        if II < Npsrs:
            RIJ[II,II,:] = np.sqrt(spectra[II].Tf**2*get_Tspan([spectra[II]])/Tspan)
            II += 1

    gwb = red_noise_powerlaw(A=A_gwb, gamma=gamma_gwb, freqs=freqs)
    der_psd_log10A = 2*np.log(10) * gwb
    der_psd_gamma = np.log(fyr/freqs) * gwb

    NcalhIJ_gamma = der_psd_gamma * RIJ
    NcalhIJ_log10A = der_psd_log10A * RIJ
    NcalInvIJ = get_NcalInv_IJ(spectra, rn_psrs)

    F_fastPTA = np.zeros((2, 2), dtype=np.float64)
    F_fastPTA[0,0] = np.sum(np.einsum('ijf, klf, jkf, lif->f', NcalInvIJ, NcalInvIJ, NcalhIJ_gamma, NcalhIJ_gamma))
    F_fastPTA[1,1] = np.sum(np.einsum('ijf, klf, jkf, lif->f', NcalInvIJ, NcalInvIJ, NcalhIJ_log10A, NcalhIJ_log10A))
    F_fastPTA[0,1] = np.sum(np.einsum('ijf, klf, jkf, lif->f', NcalInvIJ, NcalInvIJ, NcalhIJ_gamma, NcalhIJ_log10A))
    F_fastPTA[1,0] = np.sum(np.einsum('ijf, klf, jkf, lif->f', NcalInvIJ, NcalInvIJ, NcalhIJ_log10A, NcalhIJ_gamma))

    return F_fastPTA


def get_FIM_TMM(spectra:list, gamma_gwb:float, A_gwb:float, rn_psrs:dict):
    r"""
    Fisher information matrix from utilizing the timing model marginalized inverse covariance matrix.
    """

    Npsrs = len(spectra)
    psr_idx = np.arange(Npsrs)
    pairs = list(it.combinations(psr_idx,2))
    freqs = spectra[0].freqs
    Tspan = get_Tspan(spectra)

    G_dims = []
    for I in range(Npsrs):
        G_dims.append(spectra[I].G.shape[1])

    gwb = red_noise_powerlaw(A=A_gwb, gamma=gamma_gwb, freqs=freqs)
    der_psd_log10A = 2*np.log(10) * gwb
    der_psd_gamma = np.log(fyr/freqs) * gwb

    KhIJ_block_gam = np.empty((Npsrs, Npsrs), dtype=object)
    KhIJ_block_log10A = np.empty((Npsrs, Npsrs), dtype=object)

    II = 0
    for I, J in pairs:
        Chi_IJ = HD([spectra[I].phi, spectra[J].phi], [spectra[I].theta, spectra[J].theta])
        
        #Signal covariance matrix for single pulsar pair, strictly for the off-diagonal terms
        Ch_IJ_log10A = Chi_IJ * corr_from_psdIJ(freqs=freqs, psd=der_psd_log10A, toasI=spectra[I].toas, toasJ=spectra[J].toas)
        #must use trapz integration due to instability of integral 
        Ch_IJ_gam = Chi_IJ * corr_from_psdIJ(freqs=freqs, psd=der_psd_gamma, toasI=spectra[I].toas, toasJ=spectra[J].toas, fast=False)

        Kh_off_diag_log10A = jnp.matmul(spectra[I].G.T, jnp.matmul(Ch_IJ_log10A, spectra[J].G))
        Kh_off_diag_gam = jnp.matmul(spectra[I].G.T, jnp.matmul(Ch_IJ_gam, spectra[J].G))
        
        KhIJ_block_gam[I,J] = Kh_off_diag_gam
        KhIJ_block_gam[J,I] = Kh_off_diag_gam.T

        KhIJ_block_log10A[I,J] = Kh_off_diag_log10A
        KhIJ_block_log10A[J,I] = Kh_off_diag_log10A.T
    
        #conditional built into using pairs to also do the diagonal entries
        if II < Npsrs:
            Ch_gam = corr_from_psd(freqs=freqs, psd=der_psd_gamma,
                                toas=spectra[II].toas, fast=False)
            Ch_log10A = corr_from_psd(freqs=freqs, psd=der_psd_log10A,
                                toas=spectra[II].toas)
            
            KhIJ_block_gam[II,II] = jnp.matmul(spectra[II].G.T, jnp.matmul(Ch_gam, spectra[II].G))
            KhIJ_block_log10A[II,II] = jnp.matmul(spectra[II].G.T, jnp.matmul(Ch_log10A, spectra[II].G))
            II += 1

    

    KhIJ_gam = np.zeros((sum(G_dims), sum(G_dims)), dtype=np.float64)
    KhIJ_log10A = np.zeros((sum(G_dims), sum(G_dims)), dtype=np.float64)

    row_start = 0
    for i in range(Npsrs):  
        col_start = 0
        for j in range(Npsrs): 
            KhIJ_gam[row_start:row_start + G_dims[i], col_start:col_start + G_dims[j]] = KhIJ_block_gam[i,j]
            KhIJ_block_gam[i,j] = 0

            KhIJ_log10A[row_start:row_start + G_dims[i], col_start:col_start + G_dims[j]] = KhIJ_block_log10A[i,j]
            KhIJ_block_log10A[i,j] = 0

            col_start += G_dims[j]  
        row_start += G_dims[i]


    KIJ_Inv = get_KIJ_Inv(spectra, rn_psrs)
    KIJ_Inv2 = jnp.matmul(KIJ_Inv, KIJ_Inv)  
    del KIJ_Inv

    term1_gamma = jnp.matmul(KIJ_Inv2, KhIJ_gam)
    term1_log10A = jnp.matmul(KIJ_Inv2, KhIJ_log10A)

    result_gam_block = np.empty((Npsrs, Npsrs), dtype=object)
    result_log10A_block = np.empty((Npsrs, Npsrs), dtype=object)

    row_start = 0
    for i in range(Npsrs):  
        col_start = 0
        for j in range(Npsrs):
            result_gam_block[i,j] = term1_gamma[row_start:row_start + G_dims[i], col_start:col_start + G_dims[j]]
            result_log10A_block[i,j] = term1_log10A[row_start:row_start + G_dims[i], col_start:col_start + G_dims[j]]
            col_start += G_dims[j]  
        row_start += G_dims[i]

    
    NcalInvIJ = get_NcalInv_IJ(spectra, rn_psrs)

    NcalhIJ_gamma = np.zeros((Npsrs, Npsrs, freqs.size))
    NcalhIJ_log10A = np.zeros((Npsrs, Npsrs, freqs.size))
    II = 0

    for I,J in pairs:
        GtildeI = np.zeros((freqs.size, spectra[I].G.shape[1]),dtype='complex128')
        GtildeI = np.dot(np.exp(1j*2*np.pi*freqs[:,np.newaxis]*spectra[I].toas),spectra[I].G)

        GtildeJ = np.zeros((freqs.size, spectra[J].G.shape[1]),dtype='complex128')
        GtildeJ = np.dot(np.exp(1j*2*np.pi*freqs[:,np.newaxis]*spectra[J].toas),spectra[J].G)

        result_gammaIJ = jnp.matmul(jnp.conjugate(GtildeI),jnp.matmul(result_gam_block[I,J],GtildeJ.T)) / 2
        result_gammaJI = jnp.conjugate(result_gammaIJ).T

        NcalhIJ_gamma[I,J] = np.real(np.diag(result_gammaIJ)) / Tspan * 1/(NcalInvIJ[I,J]**2)
        NcalhIJ_gamma[J,I] = np.real(np.diag(result_gammaJI)) / Tspan * 1/(NcalInvIJ[J,I]**2)

        result_log10AIJ = jnp.matmul(jnp.conjugate(GtildeI),jnp.matmul(result_log10A_block[I,J],GtildeJ.T)) / 2
        result_log10AJI = jnp.conjugate(result_log10AIJ).T

        NcalhIJ_log10A[I,J] = np.real(np.diag(result_log10AIJ)) / Tspan * 1/(NcalInvIJ[I,J]**2)
        NcalhIJ_log10A[J,I] = np.real(np.diag(result_log10AJI)) / Tspan * 1/(NcalInvIJ[J,I]**2)


        if II < Npsrs:
            GtildeII = np.zeros((freqs.size, spectra[II].G.shape[1]),dtype='complex128')
            GtildeII = np.dot(np.exp(1j*2*np.pi*freqs[:,np.newaxis]*spectra[II].toas),spectra[II].G)

            result_gammaII = jnp.matmul(jnp.conjugate(GtildeII),jnp.matmul(result_gam_block[II,II],GtildeII.T)) / 2
            NcalhIJ_gamma[II,II] = np.real(np.diag(result_gammaII)) / Tspan * 1/(NcalInvIJ[II,II]**2)
            
            result_log10AII = jnp.matmul(jnp.conjugate(GtildeII),jnp.matmul(result_log10A_block[II,II],GtildeII.T)) / 2
            NcalhIJ_log10A[II,II] = np.real(np.diag(result_log10AII)) / Tspan * 1/(NcalInvIJ[II,II]**2)
        
            II+=1 

    F_TMM_hsen = np.zeros((2, 2), dtype=np.float64)
    F_TMM_hsen[0,0] = np.sum(np.einsum('ijf, klf, jkf, lif->f', NcalInvIJ, NcalInvIJ, NcalhIJ_gamma, NcalhIJ_gamma))
    F_TMM_hsen[1,1] = np.sum(np.einsum('ijf, klf, jkf, lif->f', NcalInvIJ, NcalInvIJ, NcalhIJ_log10A, NcalhIJ_log10A))
    F_TMM_hsen[0,1] = np.sum(np.einsum('ijf, klf, jkf, lif->f', NcalInvIJ, NcalInvIJ, NcalhIJ_gamma, NcalhIJ_log10A))
    F_TMM_hsen[1,0] = np.sum(np.einsum('ijf, klf, jkf, lif->f', NcalInvIJ, NcalInvIJ, NcalhIJ_log10A, NcalhIJ_gamma))

    return F_TMM_hsen

def get_FIM_TMM_diag_approx(spectra:list, gamma_gwb:float, A_gwb:float, rn_psrs:dict):
    """Fisher information matrix from the timing model marginalized inverse covariance as a diagonal approximation
    """
    Npsrs = len(spectra)
    psr_idx = np.arange(Npsrs)
    pairs = list(it.combinations(psr_idx,2))
    freqs = spectra[0].freqs
    Tspan = get_Tspan(spectra)

    gwb = red_noise_powerlaw(A=A_gwb, gamma=gamma_gwb, freqs=freqs)
    der_psd_log10A = 2*np.log(10) * gwb
    der_psd_gamma = np.log(fyr/freqs) * gwb

    KhIJ_block_gamma = []
    KhIJ_block_log10A = []

    G_dims = []
    for i in range(Npsrs):
        G_dims.append(spectra[i].G.shape[1])

    for i in range(Npsrs):
        Ch_log10A = corr_from_psd(freqs=freqs, psd=der_psd_log10A, toas=spectra[i].toas)    
        Ch_gamma = corr_from_psd(freqs=freqs, psd=der_psd_gamma, toas=spectra[i].toas, fast=False)

        Kh_log10A_val = jnp.matmul(spectra[i].G.T, jnp.matmul(Ch_log10A, spectra[i].G))
        Kh_gam_val = jnp.matmul(spectra[i].G.T, jnp.matmul(Ch_gamma, spectra[i].G))

        KhIJ_block_gamma.append(Kh_gam_val)
        KhIJ_block_log10A.append(Kh_log10A_val)
        
    KIJ_Inv = get_KII_Inv(spectra, rn_psrs)
    KIJ_Inv2 = jnp.matmul(KIJ_Inv, KIJ_Inv)
    del KIJ_Inv  
    
    KIJ_Inv2_block = []
    row_start = 0
    for i in range(Npsrs):
        KIJ_Inv2_block.append(KIJ_Inv2[row_start:row_start + G_dims[i], row_start:row_start + G_dims[i]])
        row_start += G_dims[i] 
    del KIJ_Inv2

    NcalhIJ_gamma = []
    NcalhIJ_log10A = []
    NcalInv_II = []
    for i in range(Npsrs):
        term1_gamma = jnp.matmul(KIJ_Inv2_block[i], KhIJ_block_gamma[i])
        term1_log10A = jnp.matmul(KIJ_Inv2_block[i], KhIJ_block_log10A[i])
        KIJ_Inv2_block[i] = 0
        GtildeI = np.zeros((freqs.size, spectra[i].G.shape[1]),dtype='complex128')
        GtildeI = np.dot(np.exp(1j*2*np.pi*freqs[:,np.newaxis]*spectra[i].toas),spectra[i].G)

        result_gamma = jnp.matmul(jnp.conjugate(GtildeI),jnp.matmul(term1_gamma,GtildeI.T)) / 2
        result_log10A = jnp.matmul(jnp.conjugate(GtildeI),jnp.matmul(term1_log10A,GtildeI.T)) / 2
        
        NcalInvI = spectra[i].NcalInv * get_Tspan([spectra[i]]) / Tspan

        result_1_gamma = np.real(np.diag(result_gamma)) / Tspan * 1/(NcalInvI**2)
        result_1_log10A = np.real(np.diag(result_log10A)) / Tspan * 1/(NcalInvI**2)

        NcalhIJ_gamma.append(result_1_gamma)
        NcalhIJ_log10A.append(result_1_log10A)
        NcalInv_II.append(NcalInvI) 
    del KIJ_Inv2_block


    F_TMM_hsen = np.zeros((2, 2), dtype=np.float64)
    for i in range(Npsrs):
        F_TMM_hsen[0,0] = np.sum(NcalInv_II[i]*NcalhIJ_gamma[i]*NcalInv_II[i]*NcalhIJ_gamma[i])
        F_TMM_hsen[0,1] = np.sum(NcalInv_II[i]*NcalhIJ_gamma[i]*NcalInv_II[i]*NcalhIJ_log10A[i])
        F_TMM_hsen[1,1] = np.sum(NcalInv_II[i]*NcalhIJ_log10A[i]*NcalInv_II[i]*NcalhIJ_log10A[i])
        F_TMM_hsen[1,0] = F_TMM_hsen[0,1] 
    return F_TMM_hsen


def get_FIM_fastPTA_diag_approx(spectra:list, gamma_gwb:float, A_gwb:float, rn_psrs:dict):
    """Fisher information matrix from Babak et. al. using the diagonal approximation
    """
    Npsrs = len(spectra)
    psr_idx = np.arange(Npsrs)
    pairs = list(it.combinations(psr_idx,2))
    freqs = spectra[0].freqs
    Tspan = get_Tspan(spectra)
    
    NcalInvII_TMM = np.zeros((Npsrs, Npsrs, freqs.size), dtype=np.float64)

    for i in range(Npsrs):
        NcalInvII_TMM[i,i] = spectra[i].NcalInv * get_Tspan([spectra[i]])/Tspan


    RIJ = np.zeros((Npsrs, Npsrs, freqs.size), dtype=np.float64)

    II = 0
    for I, J in pairs:
        #spectra pulsars
        
        TIJ = min(get_Tspan([spectra[I]]), get_Tspan([spectra[J]]))
        Chi_IJ = HD([spectra[I].phi, spectra[J].phi], [spectra[I].theta, spectra[J].theta])

        #response tensors
        RIJ[I,J,:] = Chi_IJ* np.sqrt(spectra[I].Tf*spectra[J].Tf*TIJ/Tspan)
        RIJ[J,I,:] = RIJ[I,J,:]

        #conditional for diagonal elements
        if II < Npsrs:
            RIJ[II,II,:] = np.sqrt(spectra[II].Tf**2*get_Tspan([spectra[II]])/Tspan)
            II += 1

    gwb = red_noise_powerlaw(A=A_gwb, gamma=gamma_gwb, freqs=freqs)
    der_psd_log10A = 2*np.log(10) * gwb
    der_psd_gamma = np.log(fyr/freqs) * gwb

    NcalhIJ_gamma = der_psd_gamma * RIJ
    NcalhIJ_log10A = der_psd_log10A * RIJ

    F_fastPTA = np.zeros((2, 2), dtype=np.float64)
    F_fastPTA[0,0] = np.sum(np.einsum('ijf, klf, jkf, lif->f', NcalInvII_TMM,  NcalInvII_TMM, NcalhIJ_gamma, NcalhIJ_gamma))
    F_fastPTA[1,1] = np.sum(np.einsum('ijf, klf, jkf, lif->f',  NcalInvII_TMM, NcalInvII_TMM, NcalhIJ_log10A, NcalhIJ_log10A))
    F_fastPTA[0,1] = np.sum(np.einsum('ijf, klf, jkf, lif->f',  NcalInvII_TMM, NcalInvII_TMM, NcalhIJ_gamma, NcalhIJ_log10A))
    F_fastPTA[1,0] = np.sum(np.einsum('ijf, klf, jkf, lif->f',  NcalInvII_TMM,  NcalInvII_TMM, NcalhIJ_log10A, NcalhIJ_gamma))

    return F_fastPTA


def get_var_hc(sens_obj, fim, gamma_gwb_mean, log10A_gwb_mean):
    """_summary_: computes variance in characteristic strain using error propagation
    """
    partial_hc_gamma, partial_hc_log10A = sens_obj.partial_hc()

    model_cov = jnp.linalg.inv(fim)
    sigma_frac_squared_gamma = model_cov[0,0]
    sigma_frac_squared_log10A = model_cov[1,1]
    var_frac_gamma_log10A = model_cov[0,1]

    sigma_gamma = np.sqrt(sigma_frac_squared_gamma) * gamma_gwb_mean
    sigma_log10A = np.sqrt(sigma_frac_squared_log10A) * log10A_gwb_mean
    var_gamma_log10A = var_frac_gamma_log10A *gamma_gwb_mean * log10A_gwb_mean

    var_hc = (partial_hc_gamma*sigma_gamma)**2 + (partial_hc_log10A*sigma_log10A)**2 + 2*partial_hc_gamma*partial_hc_log10A*var_gamma_log10A

    return var_hc            
            

def get_partial_SInv(psr, param, gamma_gwb, A_gwb, freqs):
    """Partial derivative with respect to model parameters of the the noise psd
    """
    gwb_psd = red_noise_powerlaw(A=A_gwb, gamma=gamma_gwb, freqs=freqs)

    if param == 'log10A':
        der_psd =  gwb_psd * 2 * np.log(10) 

    elif param == 'gamma':
        der_psd = gwb_psd * np.log(fyr/freqs)

    else:
        raise AttributeError("Params must be gamma or log10A")
    
    #derivative of the time-domain signal covariance matrix, and must be slow due to inverse stability
    Ch_der = corr_from_psd(psd = der_psd, freqs=freqs, toas = psr.toas, fast=False)

    K_der = jnp.matmul(psr.G.T, jnp.matmul(Ch_der, psr.G))

    K = jnp.matmul(psr.G.T, jnp.matmul(psr.N, psr.G))
    K_inv = jnp.linalg.inv(K)
    K_inv_squared = jnp.matmul(K_inv, K_inv)
    del K_inv

    result = jnp.matmul(K_inv_squared, K_der)

    Gtilde = np.zeros((freqs.size,psr.G.shape[1]),dtype='complex128')
    Gtilde = np.dot(np.exp(1j*2*np.pi*freqs[:,np.newaxis]*psr.toas),psr.G)


    TfN = jnp.matmul(np.conjugate(Gtilde),jnp.matmul(result,Gtilde.T)) / 2
    
    return -np.real(np.diag(TfN)) / get_Tspan([psr]) * resid_response(freqs=freqs)


@partial(jax.jit, static_argnames=['full_matrix', 'return_Gtilde_Ncal'])
def get_NcalInv_RRF(K_inv: jax.Array, G: jax.Array, phi:jax.Array, J: jax.Array,
                    Z: jax.Array, freqs: jax.Array, toas:jax.Array, full_matrix=False, return_Gtilde_Ncal=False):
    r"""Inverse noise-weighted transmission function utilizing rank-reduced formalism and Woodbury Lemma.

    .. math::
    \mathcal{N}^{-1}(f) \equiv  \frac{1}{2T}\tilde{G}^{*} [K^{-1} - \mathcal{Z}^{T} (\varphi^{-1} + \mathcal{Z} J)^{-1} \mathcal{Z}] \tilde{G}^T

    - [\tilde{G}]_l = \sum_{k=1}^{N_{TOA}} \mathrm{exp}(i2 \pi ft_k)[G]_{k,l}
    - \mathcal{Z} \equiv J^{T} K^{-1}
    - K \equiv G^T N G
    - J \equiv G^{T} F
    """
    T = toas.max()-toas.min()
    phi_inv = jnp.linalg.inv(phi)
    del phi

    Sigma = (phi_inv + jnp.matmul(Z, J)).T
    SigmaInv = jnp.linalg.inv(Sigma)
    del Sigma
    
    Gtilde = jnp.zeros((freqs.size, G.shape[1]),dtype='complex128')
    Gtilde = jnp.dot(jnp.exp(1j*2*jnp.pi*freqs[:,jnp.newaxis]*toas),G)

    NcalInv_ = K_inv - jnp.matmul(Z.T, jnp.matmul(SigmaInv, Z))
    del SigmaInv
   
    TfN = jnp.matmul(jnp.conjugate(Gtilde),jnp.matmul(NcalInv_,Gtilde.T)) / 2
    if return_Gtilde_Ncal:
        return jnp.real(TfN), Gtilde, jnp.linalg.inv(NcalInv_)
    elif full_matrix:
        return jnp.real(TfN)
    else:
        return jnp.real(jnp.diag(TfN)) / T

def resid_response(freqs):
    r"""
    Returns the timing residual response function for a pulsar across as set of
    frequencies. See Equation (53) in `[1]`_.

    .. math::
        \mathcal{R}(f)=\frac{1}{12\pi^2\;f^2}

    .. _[1]: https://arxiv.org/abs/1907.04341
    """
    return 1/(12 * np.pi**2 * freqs**2)


class Pulsar(object):
    """
    Class to encode information about individual pulsars.

    Parameters
    ----------

    toas : array
        Pulsar Times of Arrival [sec].

    toaerrs : array
        Pulsar TOA errors [sec].

    phi : float
        Ecliptic longitude of pulsar [rad].

    theta : float
        Ecliptic latitude of pulsar [rad].

    name: str
        name of pulsar. attempts to name pulsar based off phi, theta.
        default is 'J0000+0000'.

    designmatrix : array
        Design matrix for pulsar's timing model. N_TOA x N_param.

    N : array
        Covariance matrix for the pulsar. N_TOA x N_TOA. Made from toaerrs
        if not provided.

    pdist : astropy.quantity, float
        Earth-pulsar distance. Default units is kpc.

    """
    def __init__(self, toas, toaerrs, phi=None, theta=None, name=None,
                 designmatrix=None, N=None, pdist=1.0*u.kpc, A_rn=None,
                 alpha=None,):
        self.toas = toas
        self.toaerrs = toaerrs
        self.phi = phi
        self.theta = theta
        self.pdist = make_quant(pdist,'kpc')
        self.A_rn = A_rn
        self.alpha = alpha
        self.name = name

        if name is None:
            try:
                self.name = skycoord_to_Jname(theta_phi_to_SkyCoord(theta,phi))
            except:
                self.name = 'J0000+0000'
        else:
            self.name = str(name)
        
        if N is None:
            self.N = np.diag(toaerrs**2) #N ==> weights
        else:
            self.N = N

        if designmatrix is None:
            self.designmatrix = create_design_matrix(toas, RADEC=True,
                                                     PROPER=True, PX=True)
        else:
            self.designmatrix = designmatrix

    def filter_data(self, start_time=None, end_time=None):
        """
        Parameters
        ==========
        start_time - float
            MJD at which to begin data subset.
        end_time - float
            MJD at which to end data subset.

        Filter data to create a time-slice of overall dataset.
        Function adapted from enterprise.BasePulsar() class.
        """
        if start_time is None and end_time is None:
            mask = np.ones(self.toas.shape, dtype=bool)
        else:
            mask = np.logical_and(self.toas >= start_time * 86400, self.toas <= end_time * 86400)

        self.toas = self.toas[mask]
        self.toaerrs = self.toaerrs[mask]
        self.N = self.N[mask, :][:, mask]

        self.designmatrix = create_design_matrix(self.toas, RADEC=True, PROPER=True, PX=True)
        #self.designmatrix = self.designmatrix[mask, :]
        #dmx_mask = np.sum(self.designmatrix, axis=0) != 0.0
        #self.designmatrix = self.designmatrix[:, dmx_mask]
        self._G = G_matrix(designmatrix=self.designmatrix)

    def change_cadence(self, start_time=0, end_time=1_000_000,
                       cadence=None, cadence_factor=4, uneven=False, 
                       A_gwb=None, alpha_gwb=-2/3., freqs=None,
                       fast=True,):
        """
        Parameters
        ==========
        start_time - float
            MJD at which to begin altered cadence.
        end_time - float
            MJD at which to end altered cadence.
        cadence - float
            cadence for the modified campaign [toas/year]
        cadence_facter - float
            (instead of cadence) factor by which to modify the old cadence.
        uneven - bool
            whether or not to evenly space observation epochs
        A_gwb - float
            amplitude of injected gwb self-noise
        alpha_gwb - float
            spectral index of injected gwb self-noise.
            note that this is residual space spectral index.
        freqs - array
            frequencies to construct the gwb noise and intrinsic noise
        fast - bool
            faster but slightly less accurate method to calculate noise injected in N.

        Change observing cadence in a given time range.
        Recalculate pulsar noise properties.
        """
        mask_before = self.toas <= start_time * 86400
        mask_after = self.toas >= end_time * 86400
        old_Ntoas = np.sum(
                    np.logical_and(self.toas >= start_time * 86400,
                                    self.toas <= end_time * 86400)
                )
        # store the old toas and errors
        old_toas = self.toas
        old_toaerrs = self.toaerrs
        # calculate old cadence then modified cadence
        if start_time < min(old_toas)/84600:
            start_time = min(old_toas)/84600
        if end_time < min(old_toas)/84600:
            print("trying to change non-existant campaign")
            return 0
        duration = end_time - start_time # in MJD
        old_cadence = old_Ntoas / duration * 365.25 # cad is Ntoas/year
        if cadence is not None:
            new_cadence = cadence
        else:
            new_cadence = old_cadence * cadence_factor
        # create new toas and toa errors
        campaign_Ntoas = int(np.floor( duration / 365.25 * new_cadence ))
        campaign_toas = np.linspace(start_time, end_time, campaign_Ntoas) * 86400
        if uneven:
            # FIXME check this with jeff to see what he was going for
            # in sim_pta()
            dt = duration / campaign_Ntoas / 8 * yr_sec
            campaign_toas += np.random.uniform(-dt, dt, size=campaign_Ntoas)
        self.toas = np.concatenate([old_toas[mask_before], campaign_toas, old_toas[mask_after]])
        campaign_toaerrs = np.median(old_toaerrs)*np.ones(campaign_Ntoas)
        # TODO can only use a fixed toaerr for the duration of the campaign
        #self.toaerrs = np.concatenate([old_toaerrs[mask_before], campaign_toaerrs, old_toaerrs[mask_after]])
        self.toaerrs = np.ones(len(self.toas))*old_toaerrs[0]
        print(f"old: {len(old_toaerrs)}, new: {len(self.toaerrs)}")
        # recalculate N, designmatrix, G with new toas
        N = np.diag(self.toaerrs**2)
        if self.A_rn is not None:
            plaw = red_noise_powerlaw(A=self.A_rn,
                                      alpha=self.alpha,
                                      freqs=freqs)
            N += corr_from_psd(freqs=freqs, psd=plaw, toas=self.toas, fast=fast)

        if A_gwb is not None:
            gwb = red_noise_powerlaw(A=A_gwb,
                                     alpha=alpha_gwb,
                                     freqs=freqs)
            N += corr_from_psd(freqs=freqs, psd=gwb, toas=self.toas, fast=fast)
        self.designmatrix = create_design_matrix(self.toas, RADEC=True, PROPER=True, PX=True)
        self._G = G_matrix(designmatrix=self.designmatrix)
        self.N = N

    def change_sigma(self, start_time=0, end_time=1_000_000,
                       new_sigma=None, sigma_factor=4, uneven=False, 
                       A_gwb=None, alpha_gwb=-2/3., freqs=None,
                       fast=True,):
        """
        Parameters
        ==========
        start_time - float
            MJD at which to begin altered toa errors.
        end_time - float
            MJD at which to end altered toa errors.
        new_sigma - float
            uncertainty of toas for the modified campaign [microseconds]
        sigma_facter - float
            (instead of sigmas) factor by which to modify the campaign sigmas.
        uneven - bool
            whether or not to evenly space observation epochs
        A_gwb - float
            amplitude of injected gwb self-noise
        alpha_gwb - float
            spectral index of injected gwb self-noise.
            note that this is residual space spectral index (alpha).
        freqs - array
            frequencies to construct the gwb noise and intrinsic noise
        fast - bool
            faster but slightly less accurate method to calculate noise injected in N.

        Change observing cadence in a given time range.
        Recalculate pulsar noise properties.
        """
        mask_before = self.toas <= start_time * 86400
        mask_after = self.toas >= end_time * 86400
        campaign_mask = np.logical_and(self.toas >= start_time * 86400,
                                    self.toas <= end_time * 86400)
        campaign_ntoas = np.sum(campaign_mask)
        # store the old toa errors
        toaerrs_campaign = self.toaerrs[campaign_mask]
        # modify the campaign toa errors
        if sigma_factor is not None:
            toaerrs_campaign = sigma_factor * toaerrs_campaign
        elif sigma_factor is None and new_sigma is not None:
            toaerrs_campaign = np.ones(campaign_ntoas)*new_sigma
        self.toaerrs = np.concatenate([self.toaerrs[mask_before], toaerrs_campaign, self.toaerrs[mask_after]])
        # recalculate N, designmatrix, G with new toas
        N = np.diag(self.toaerrs**2)
        if self.A_rn is not None:
            plaw = red_noise_powerlaw(A=self.A_rn,
                                      alpha=self.alpha,
                                      freqs=freqs)
            N += corr_from_psd(freqs=freqs, psd=plaw, toas=self.toas, fast=fast)

        if A_gwb is not None:
            gwb = red_noise_powerlaw(A=A_gwb,
                                     alpha=alpha_gwb,
                                     freqs=freqs)
            N += corr_from_psd(freqs=freqs, psd=gwb, toas=self.toas, fast=fast)
        self.designmatrix = create_design_matrix(self.toas, RADEC=True, PROPER=True, PX=True)
        self._G = G_matrix(designmatrix=self.designmatrix)
        self.N = N

    def psr_h5(self, dir: str, compress_val: int = 0):
        """Writes Pulsar object to HDF5 files

        Args:
            - dir (str): directory of HDF5 file
            - compress_val: gzip compression value, ranges from  0 to 9 with
              0 yielding no compression. Only large arrays such as G, N, and 
              designmatrix are compressed.
        """
        with h5py.File(dir, 'a') as f:
            hdf5_psr = f.create_group(self.name)
            hdf5_psr.create_dataset('toas', self.toas.shape, self.toas.dtype, data=self.toas)
            hdf5_psr.create_dataset('toaerrs', self.toaerrs.shape, self.toaerrs.dtype, data=self.toaerrs)
            hdf5_psr.create_dataset('phi', (1,), float, data=self.phi)
            hdf5_psr.create_dataset('theta', (1,), float, data=self.theta)
            hdf5_psr.create_dataset('designmatrix', self.designmatrix.shape, self.designmatrix.dtype, data=self.designmatrix, 
                                    compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('G', self.G.shape, self.G.dtype, data=self.G, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('N', self.N.shape, self.N.dtype, data=self.N, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('pdist', (2,), float, data=self.pdist)
            f.flush()
    
    @property
    def G(self):
        """Timing Model Projection Matrix."""
        if not hasattr(self, '_G'):
            self._G = G_matrix(designmatrix=self.designmatrix)
        return self._G
    
    @cached_property
    def K_inv(self):
        """Timing Model Marginalized Inverse White Noise Covariance Matrix."""
        L = jsc.linalg.cholesky(self.N)        
        A = jnp.matmul(L,self.G)
        del L
        K = jnp.matmul(A.T,A)
        del A
        return jnp.linalg.inv(K)

class Spectrum(object):
    """Class to encode the spectral information for a single pulsar.

    Parameters
    ----------

    psr : `hasasia.Pulsar`
        A `hasasia.Pulsar` instance.

    nf : int, optional
        Number of frequencies over which to build the various spectral
        densities.

    fmin : float, optional [Hz]
        Minimum frequency over which to build the various spectral
        densities. Defaults to the timespan/5 of the pulsar.

    fmax : float, optional [Hz]
        Minimum frequency over which to build the various spectral
        densities.

    freqs : array, optional [Hz]
        Optionally supply an array of frequencies over which to build the
        various spectral densities.
    """
    def __init__(self, psr, A_gwb, gamma_gwb, nf=400, fmin=None, fmax=2e-7,
                 freqs=None, tm_fit=True, **Tf_kwargs):
        self._H_0 = 72 * u.km / u.s / u.Mpc
        self.toas = psr.toas
        self.toaerrs = psr.toaerrs
        self.phi = psr.phi
        self.theta = psr.theta
        self.name = psr.name  
        
        if hasattr(psr, 'N'):
            self.N = psr.N
        else:
            self.K_inv = psr.K_inv

        self.gamma_gwb = gamma_gwb
        self.A_gwb = A_gwb

        self.G = psr.G
        self.designmatrix = psr.designmatrix
        self.pdist = psr.pdist
        self.tm_fit = tm_fit
        self.Tf_kwargs = Tf_kwargs

        try:
            self.name = psr.name
        except AttributeError:
            self.name = 'J0000+0000'

        if freqs is None:
            f0 = 1 / get_Tspan([psr])
            if fmin is None:
                fmin = f0/5
            self.freqs = np.logspace(np.log10(fmin), np.log10(fmax), nf)
        else:
            self.freqs = freqs

        self._psd_prefit = np.zeros_like(self.freqs)

    @property
    def psd_postfit(self):
        """Postfit Residual Power Spectral Density"""
        if not hasattr(self, '_psd_postfit'):
            self._psd_postfit = self.psd_prefit * self.NcalInv
        return self._psd_postfit

    @property
    def psd_prefit(self):
        """Prefit Residual Power Spectral Density"""
        if np.all(self._psd_prefit==0):
            raise ValueError('Must set Prefit Residual Power Spectral Density.')
            # print('No Prefit Residual Power Spectral Density set.\n'
            #       'Setting psd_prefit to harmonic mean of toaerrs.')
            # sigma = sps.hmean(self.toaerrs)
            # dt = 14*24*3600 # 2 Week Cadence
            # self.add_white_noise_pow(sigma=sigma,dt=dt)

        return self._psd_prefit

    @property
    def Tf(self):
        """Transmission function"""
        if not hasattr(self, '_Tf'):
            self._Tf,_,_ = get_Tf(designmatrix=self.designmatrix,
                                  toas=self.toas, N=self.N,
                                  freqs=self.freqs, from_G=True, Gmatrix=self.G,
                                  **self.Tf_kwargs)
        return self._Tf


    @property
    def NcalInv(self):
        """Inverse Noise Weighted Transmission Function."""
        if not hasattr(self, '_NcalInv'):
            self._NcalInv = get_NcalInv(psr=self, freqs=self.freqs,
                                        tm_fit=self.tm_fit, Gmatrix=self.G)
        return self._NcalInv

    @property
    def P_n(self):
        """Inverse Noise Weighted Transmission Function."""
        if not hasattr(self, '_P_n'):
            self._P_n = np.power(get_NcalInv(psr=self, freqs=self.freqs,
                                             tm_fit=False), -1)
        return self._P_n

    @property
    def S_I(self):
        r"""Strain power sensitivity for this pulsar. Equation (74) in `[1]`_

        .. math::
            S_I=\frac{1}{\mathcal{N}^{-1}\;\mathcal{R}}

        .. _[1]: https://arxiv.org/abs/1907.04341
        """
        if not hasattr(self, '_S_I'):
            self._S_I = 1/resid_response(self.freqs)/self.NcalInv
        return self._S_I
    
    @property
    def partial_gamma_SI_Inv(self):
        """partial derivative of the noise psd with respect to gamma
        """
        if not hasattr(self, '_partial_gamma_SI_Inv'):
            self._partial_gamma_SI_Inv = get_partial_SInv(self, param='gamma', gamma_gwb=self.gamma_gwb, A_gwb = self.A_gwb, freqs=self.freqs)
        return self._partial_gamma_SI_Inv

    @property
    def partial_log10A_SI_Inv(self):
        """partial derivative of the noise psd with respect to log10A
        """
        if not hasattr(self, '_partial_log10A_SI_Inv'):
            self._partial_log10A_SI_Inv = get_partial_SInv(self, param='log10A', gamma_gwb=self.gamma_gwb, A_gwb = self.A_gwb, freqs=self.freqs)
        return self._partial_log10A_SI_Inv

    @property   
    def S_R(self):
        r"""Residual power sensitivity for this pulsar.

        .. math::
            S_R=\frac{1}{\mathcal{N}^{-1}}

        """
        if not hasattr(self, '_S_R'):
            self._S_R = 1/self.NcalInv
        return self._S_R

    @property
    def h_c(self):
        r"""Characteristic strain sensitivity for this pulsar.

        .. math::
            h_c=\sqrt{f\;S_I}
        """
        if not hasattr(self, '_h_c'):
            self._h_c = np.sqrt(self.freqs * self.S_I)
        return self._h_c

    @property
    def Omega_gw(self):
        r"""Energy Density sensitivity.

        .. math::
            \Omega_{gw}=\frac{2\pi^2}{3\;H_0^2}f^3\;S_I
        """
        self._Omega_gw = ((2*np.pi**2/3) * self.freqs**3 * self.S_I
                           / self._H_0.to('Hz').value**2)
        return self._Omega_gw

    def add_white_noise_power(self, sigma=None, dt=None, vals=False):
        r"""
        Add power law red noise to the prefit residual power spectral density.

        **Note:** All noise information is furnished by the covariance matrix in
        the `hasasia.Pulsar` object, this is simply useful for bookkeeping and
        plots.

        Parameters
        ----------
        sigma : float
            TOA error.

        dt : float
            Time between observing epochs in [seconds].

        vals : bool
            Whether to return the psd values as an array. Otherwise just added
            to `self.psd_prefit`.
        """
        white_noise = 2.0 * dt * (sigma)**2 * np.ones_like(self.freqs)
        self._psd_prefit += white_noise
        if vals:
            return white_noise

    def add_red_noise_power(self, A=None, gamma=None, vals=False):
        r"""
        Add power law red noise to the prefit residual power spectral density.
        As :math:`P=A^2(f/fyr)^{-\gamma}`.

        **Note:** All noise information is furnished by the covariance matrix in
        the `hasasia.Pulsar` object, this is simply useful for bookkeeping and
        plots.

        Parameters
        ----------
        A : float
            Amplitude of red noise.

        gamma : float
            Spectral index of red noise powerlaw.

        vals : bool
            Whether to return the psd values as an array. Otherwise just added
            to `self.psd_prefit`.
        """
        ff = self.freqs
        red_noise = A**2*(ff/fyr)**(-gamma)/(12*np.pi**2) * yr_sec**3
        self._psd_prefit += red_noise
        if vals:
            return red_noise

    def add_noise_power(self,noise):
        r"""Add any spectrum of noise. Must match length of frequency array.

        **Note:** All noise information is furnished by the covariance matrix in
        the `hasasia.Pulsar` object, this is simply useful for bookkeeping and
        plots.
        """
        self._psd_prefit += noise

    def spec_h5(self, dir:str, compress_val: int = 0):
        """Writes hasasia Spectrum object to hdf5 file
        
        Args:
        - psr (hasasia.Spectrum): pulsar spectrum object
        - dir (str): directory in which to save pulsar object. 
        - compress_val (int): compression value ranging from 0 to 9.
        """  
        with h5py.File(dir, 'a') as f:
            hdf5_psr = f.create_group(self.name)
            hdf5_psr.create_dataset('toas', self.toas.shape, self.toas.dtype, data=self.toas, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('freqs', self.freqs.shape,self.freqs.dtype, data=self.freqs, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('phi', (1,), float, data=self.phi)
            hdf5_psr.create_dataset('theta', (1,), float, data=self.theta)
            hdf5_psr.create_dataset('NcalInv', self.NcalInv.shape, self.NcalInv.dtype, data=self.NcalInv, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('S_I', self.S_I.shape, self.S_I.dtype, data=self.S_I, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('G', self.G.shape, self.G.dtype, data=self.G, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('N', self.N.shape, self.N.dtype, data=self.N, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('Tf', self.Tf.shape, self.Tf.dtype, data=self.Tf, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('toaerrs', self.toaerrs.shape,self.toaerrs.dtype, data=self.toaerrs, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('pdist', (2,), float, data=self.pdist)
            hdf5_psr.create_dataset('Mmat', self.designmatrix.shape, self.designmatrix.dtype, data=self.designmatrix, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('partial_gamma_SI_Inv', self.partial_gamma_SI_Inv.shape, self.partial_gamma_SI_Inv.dtype, data=self.partial_gamma_SI_Inv, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('partial_log10A_SI_Inv', self.partial_log10A_SI_Inv.shape, self.partial_log10A_SI_Inv.dtype, data=self.partial_log10A_SI_Inv, compression="gzip", compression_opts=compress_val)
            f.flush()


class Spectrum_RRF(object):
    """Class to encode the spectral information for a single pulsar for use in Rank Reduced Formalism.

    Parameters
    ----------

    psr : `hasasia.Pulsar`
        A `hasasia.Pulsar` instance.

    amp_irn : float
        Pulsar red noise spectra amplitude

    gamma_irn: float
        Pulsar red noise spectral index

    freqs_irn_comp: int
        Number of harmonics that the instrinsic red noise is present in the frequencies

    amp_gw : float
        GWB spectra amplitude

    gamma_gw: float
        GWB spectral index

    freqs_gwb_comp: int
        Number of harmonics that the GWB is present in the frequencies

    nf : int, optional
        Number of frequencies over which to build the various spectral
        densities.

    fmin : float, optional [Hz]
        Minimum frequency over which to build the various spectral
        densities. Defaults to the timespan/5 of the pulsar.

    fmax : float, optional [Hz]
        Minimum frequency over which to build the various spectral
        densities.

    freqs : array, optional [Hz]
        Optionally supply an array of frequencies over which to build the
        various spectral densities.
    """
    def __init__(self, psr:Pulsar, Tspan:float, freqs_gw_comp:int, amp_gw:float, gamma_gw:float,
                freqs_irn_comp:int, amp_irn = None, gamma_irn = None, nf=400, fmin=None,
                fmax=2e-7, freqs=None,  tm_fit=True):
        self._H_0 = 72 * u.km / u.s / u.Mpc
        self.toas = psr.toas
        self.toaerrs = psr.toaerrs
        
        self.phi = psr.phi
        self.theta = psr.theta
        self.Tspan = Tspan

        self.G = psr.G
        self.K_inv = psr.K_inv 

        self.designmatrix = psr.designmatrix
        self.pdist = psr.pdist

        if freqs_gw_comp > freqs_irn_comp:
            raise Exception('Frequencies of the GWB MUST be a subset of the intrinsic red noise frequencies.')

        #intrinsic red noise frequencies and psd parameters
        self.freqs_irn = np.linspace(1/Tspan, freqs_irn_comp/Tspan, freqs_irn_comp)
        self.amp = amp_irn
        self.gamma = gamma_irn

        #gwb frequencies and psd parameters
        self.freqs_gwb = self.freqs_irn[:freqs_gw_comp]
        self.amp_gw = amp_gw
        self.gamma_gw = gamma_gw

        self.tm_fit = tm_fit
        
        if freqs is None:
            f0 = 1 / get_Tspan([psr])
            if fmin is None:
                fmin = f0/5
            self.freqs = np.logspace(np.log10(fmin), np.log10(fmax), nf)
        else:
            self.freqs = freqs
        self._psd_prefit = np.zeros_like(self.freqs)

    def psd_postfit(self):
        """Postfit Residual Power Spectral Density"""
        if not hasattr(self, '_psd_postfit'):
            self._psd_postfit = self.psd_prefit * self.NcalInv
        return self._psd_postfit

    @property
    def psd_prefit(self):
        """Prefit Residual Power Spectral Density"""
        if np.all(self._psd_prefit==0):
            raise ValueError('Must set Prefit Residual Power Spectral Density.')
            # print('No Prefit Residual Power Spectral Density set.\n'
            #       'Setting psd_prefit to harmonic mean of toaerrs.')
            # sigma = sps.hmean(self.toaerrs)
            # dt = 14*24*3600 # 2 Week Cadence
            # self.add_white_noise_pow(sigma=sigma,dt=dt)

        return self._psd_prefit

    @cached_property
    def Cirn(self):
        """Intrinsic Red Noise Fourier Covariance Matrix"""
        nf =  self.freqs_irn.size
        #For pulsars with no intrinsic red noise, then have an extremely small amplitude psd value
        if self.gamma == None or self.amp == None:
            C_rn_proto = red_noise_powerlaw(A=1e-40, gamma=0, freqs=self.freqs_irn)
            C_rn = np.zeros((2*nf, 2*nf))
            C_rn[::2, ::2] = np.diag(C_rn_proto)   #odd elements
            C_rn[1::2, 1::2] = np.diag(C_rn_proto) #even elements
            del C_rn_proto
        else:
            #creation of fourier coeffiecent covariance matrix, and computes inverse
            C_rn_proto = red_noise_powerlaw(A=self.amp, gamma=self.gamma, freqs=self.freqs_irn)
            C_rn = np.zeros((2*nf, 2*nf))
            C_rn[::2, ::2] = np.diag(C_rn_proto)   #odd elements
            C_rn[1::2, 1::2] = np.diag(C_rn_proto) #even elements
            del C_rn_proto
        return C_rn/self.Tspan
    
    @cached_property
    def Cgw(self):
        """Gravitational Wave Fourier Covariance Matrix"""
        nf_gw = self.freqs_gwb.size
        gwb_power = red_noise_powerlaw(A=self.amp_gw, gamma=self.gamma_gw, freqs=self.freqs_gwb)
        C_gwbproto = np.zeros((2*nf_gw, 2*nf_gw))
        C_gwbproto[::2, ::2] = np.diag(gwb_power)   #odd elements
        C_gwbproto[1::2, 1::2] = np.diag(gwb_power) #even elements
        del gwb_power

        C_gwb = np.zeros((2*self.freqs_irn.size, 2*self.freqs_irn.size))
        #creating a mask to overlay the GB covariance matrix onto the IRN covariance matrix
        mask = np.full(self.freqs_irn.size, False)
        for i in range(self.freqs_irn.size):
            for j in range(self.freqs_gwb.size):
                #assumption here is that GB frequencies is a subset of IRN frequencies 
                if self.freqs_irn[i] == self.freqs_gwb[j]:
                    mask[i] = True
                    continue
        #duplicates the mask for use of 2Nfreq formalism
        mask_rp = np.repeat(mask, 2)
        del mask
        C_gwb[np.ix_(mask_rp, mask_rp)] = C_gwbproto

        return C_gwb/self.Tspan

    @cached_property
    def J(self):
        """G^T F. Common quantity used in RRF contained within the Woodbury Identity"""
        nf = self.freqs_irn.size
        N = len(self.toas)
        
        #Fourier Design matrix
        F  = jnp.zeros((N, 2 * nf))
        f = jnp.arange(1, nf + 1) / self.Tspan
        F = F.at[:, ::2].set(jnp.sin(2 * jnp.pi * self.toas[:, None] * f[None, :])) 
        F = F.at[:, 1::2].set(jnp.cos(2 * jnp.pi * self.toas[:, None] * f[None, :])) 
        del f   
        return jnp.matmul(self.G.T, F)
    

    @cached_property
    def Z(self):
        """F^T G K^{-1}. Common quantity used in RRF contained within the Woodbury Indentity"""
        return jnp.matmul(self.J.T, self.K_inv)
    

    @property
    def NcalInv(self, full_matrix=False, return_Gtilde_Ncal=False):
        """_summary_

        Args:
            full_matrix (bool, optional): _description_. Defaults to False.
            return_Gtilde_Ncal (bool, optional): _description_. Defaults to False.

        Returns:, 
            _type_: _description_
        """
        #Defining Ncal and NcalInv depending on existence of self.N or self.K_inv
        if not hasattr(self, '_NcalInv'):
            phi = jnp.array(self.Cgw + self.Cirn)
            K_inv = jnp.array(self.K_inv)
            G = jnp.array(self.G)
            J = jnp.array(self.J)
            Z = jnp.array(self.Z)
            toas = jnp.array(self.toas)
            freqs = jnp.array(self.freqs)
            self._NcalInv = get_NcalInv_RRF(K_inv, G, phi, J,
                    Z, freqs, toas, full_matrix=full_matrix, return_Gtilde_Ncal=return_Gtilde_Ncal)
        return self._NcalInv
            
    @property
    def P_n(self):
        """Inverse Noise Weighted Transmission Function."""
        if not hasattr(self, '_P_n'):
            self._P_n = np.power(self.NcalInv, -1)
        return self._P_n

    @property
    def S_I(self):
        r"""Strain power sensitivity for this pulsar. Equation (74) in `[1]`_

        .. math::
            S_I=\frac{1}{\mathcal{N}^{-1}\;\mathcal{R}}

        .. _[1]: https://arxiv.org/abs/1907.04341
        """
        if not hasattr(self, '_S_I'):
            self._S_I = 1/resid_response(self.freqs)/self.NcalInv
        return self._S_I

    @property
    def S_R(self):
        r"""Residual power sensitivity for this pulsar.

        .. math::
            S_R=\frac{1}{\mathcal{N}^{-1}}

        """
        if not hasattr(self, '_S_R'):
            self._S_R = 1/self.NcalInv
        return self._S_R

    @property
    def h_c(self):
        r"""Characteristic strain sensitivity for this pulsar.

        .. math::
            h_c=\sqrt{f\;S_I}
        """
        if not hasattr(self, '_h_c'):
            #needed to make S_I positive
            self._h_c = np.sqrt(self.freqs * self.S_I)
        return self._h_c

    @property
    def Omega_gw(self):
        r"""Energy Density sensitivity.

        .. math::
            \Omega_{gw}=\frac{2\pi^2}{3\;H_0^2}f^3\;S_I
        """
        self._Omega_gw = ((2*np.pi**2/3) * self.freqs**3 * self.S_I
                           / self._H_0.to('Hz').value**2)
        return self._Omega_gw

    def add_white_noise_power(self, sigma=None, dt=None, vals=False):
        r"""
        Add power law red noise to the prefit residual power spectral density.

        **Note:** All noise information is furnished by the covariance matrix in
        the `hasasia.Pulsar` object, this is simply useful for bookkeeping and
        plots.

        Parameters
        ----------
        sigma : float
            TOA error.

        dt : float
            Time between observing epochs in [seconds].

        vals : bool
            Whether to return the psd values as an array. Otherwise just added
            to `self.psd_prefit`.
        """
        white_noise = 2.0 * dt * (sigma)**2 * np.ones_like(self.freqs)
        self._psd_prefit += white_noise
        if vals:
            return white_noise

    def add_red_noise_power(self, A=None, gamma=None, vals=False, f_gw=None):
        r"""
        Add power law red noise to the prefit residual power spectral density.
        As :math:`P=A^2(f/fyr)^{-\gamma}`.

        **Note:** All noise information is furnished by the covariance matrix in
        the `hasasia.Pulsar` object, this is simply useful for bookkeeping and
        plots.

        Parameters
        ----------
        A : float
            Amplitude of red noise.

        gamma : float
            Spectral index of red noise powerlaw.

        vals : bool
            Whether to return the psd values as an array. Otherwise just added
            to `self.psd_prefit`.
        """
        if f_gw is None:
            ff = self.freqs
        else:
            ff = f_gw
        red_noise = A**2*(ff/fyr)**(-gamma)/(12*np.pi**2) * yr_sec**3
        if vals:
            return red_noise

    def add_noise_power(self,noise):
        r"""Add any spectrum of noise. Must match length of frequency array.

        **Note:** All noise information is furnished by the covariance matrix in
        the `hasasia.Pulsar` object, this is simply useful for bookkeeping and
        plots.
        """
        self._psd_prefit += noise

    def spec_h5(self, dir:str, compress_val: int = 0):
        """Writes hasasia Spectrum object to hdf5 file
        
        Args:
        - psr (hasasia.Spectrum): pulsar spectrum object
        - dir (str): directory in which to save pulsar object. 
        - compress_val (int): compression value ranging from 0 to 9.
        """  
        with h5py.File(dir, 'a') as f:
            hdf5_psr = f.create_group(self.name)
            hdf5_psr.create_dataset('toas', self.toas.shape, self.toas.dtype, data=self.toas, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('freqs', self.freqs.shape,self.freqs.dtype, data=self.freqs, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('phi', (1,), float, data=self.phi)
            hdf5_psr.create_dataset('theta', (1,), float, data=self.theta)
            hdf5_psr.create_dataset('NcalInv', self.NcalInv.shape, self.NcalInv.dtype, data=self.NcalInv, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('S_I', self.S_I.shape, self.S_I.dtype, data=self.S_I, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('G', self.G.shape, self.G.dtype, data=self.G, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('N', self.N.shape, self.N.dtype, data=self.N, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('Tf', self.Tf.shape, self.Tf.dtype, data=self.Tf, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('toaerrs', self.toaerrs.shape,self.toaerrs.dtype, data=self.toaerrs, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('pdist', (2,), float, data=self.pdist)
            hdf5_psr.create_dataset('Mmat', self.designmatrix.shape, self.designmatrix.dtype, data=self.designmatrix, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('partial_gamma_SI_Inv', self.partial_gamma_SI_Inv.shape, self.partial_gamma_SI_Inv.dtype, data=self.partial_gamma_SI_Inv, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('partial_log10A_SI_Inv', self.partial_log10A_SI_Inv.shape, self.partial_log10A_SI_Inv.dtype, data=self.partial_log10A_SI_Inv, compression="gzip", compression_opts=compress_val)
            f.flush()

class SensitivityCurve(object):
    r"""
    Base class for constructing PTA sensitivity curves. Takes a list of
    `hasasia.Spectrum` objects as input.
    """
    def __init__(self, spectra):

        if not isinstance(spectra, list):
            raise ValueError('Must provide list of spectra!!')

        self._H_0 = 72 * u.km / u.s / u.Mpc
        self.Npsrs = len(spectra)
        self.phis = np.array([p.phi for p in spectra])
        self.thetas = np.array([p.theta for p in spectra])
        self.Tspan = get_Tspan(spectra)
        # f0 = 1 / self.Tspan
        # if fmin is None:
        #     fmin = f0/5

        #Check to see if all frequencies are equal.
        freq_check = [sp.freqs for sp in spectra]
        if np.all(freq_check == spectra[0].freqs):
            self.freqs = spectra[0].freqs
        else:
            raise ValueError('All frequency arrays must match for sensitivity'
                             ' curve calculation!!')

        self.SnI = np.array([sp.S_I for sp in spectra])

    def to_pickle(self, filepath):
        self.filepath = filepath
        with open(filepath, "wb") as fout:
            pickle.dump(self, fout)

    def fidx(self,f):
        """Get the indices of a frequencies in the frequency array."""
        if isinstance(f, int) or isinstance(f, float):
            f = np.array([f])
            f = np.asarray(f)
        return np.array([np.argmin(abs(ff-self.freqs)) for ff in f])

    @property
    def S_eff(self):
        """Strain power sensitivity. """
        raise NotImplementedError('Effective Strain Power Sensitivity'
                                  'method must be defined.')

    @property
    def h_c(self):
        """Characteristic strain sensitivity"""
        if not hasattr(self, '_h_c'):
            self._h_c = np.sqrt(self.freqs * self.S_eff)
        return self._h_c

    @property
    def Omega_gw(self, H_0=None):
        """Energy Density sensitivity
        Default value of H_0 is 72 km/s/Mpc -- can supply different value."""
        self._Omega_gw = ((2*np.pi**2/3) * self.freqs**3 * self.S_eff
                           / self.H_0(H_0).to('Hz').value**2)
        return self._Omega_gw
   
    @property
    def hsq_Omega_gw(self, H_0=None):
        """
        Energy Density sensitivity
        Uses a common convention for energy density: h^2 * Omega_gw
        where h^2 is the dimensionless Hubble constant squared.
        Default value of H_0 is 72 km/s/Mpc -- can supply different value.
        """
        return self.Omega_gw(H_0) * (self.H_0(H_0)/(100*u.km/u.Mpc/u.s))**2

    def H_0(self, H_0=None):
        """Hubble Constant. Assumed to be in units of km /(s Mpc) unless
        supplied as an `astropy.quantity`.
        Default value of H_0 is 72 km/s/Mpc -- can supply different value."""
        if H_0 is not None:
            self._H_0 = (make_quant(H_0,'km /(s Mpc)'))
        else:
            self._H_0 = make_quant(self._H_0,'km /(s Mpc)')
        return self._H_0


class GWBSensitivityCurve(SensitivityCurve):
    r"""
    Class to produce a sensitivity curve for a gravitational wave
    background, using Hellings-Downs spatial correlations.

    Parameters
    ----------
    orf : str, optional {'hd', 'st', 'dipole', 'monopole'}
        Overlap reduction function to be used in the sensitivity curve.
        Maybe be Hellings-Downs, Scalar-Tensor, Dipole or Monopole.

    """

    def __init__(self, spectra, orf='hd',autocorr=False):

        super().__init__(spectra)
        if orf == 'hd':
            Coff = HellingsDownsCoeff(self.phis, self.thetas, autocorr=autocorr)
        elif orf == 'st':
            Coff = ScalarTensorCoeff(self.phis, self.thetas)
        elif orf == 'dipole':
            Coff = DipoleCoeff(self.phis, self.thetas)
        elif orf == 'monopole':
            Coff = MonopoleCoeff(self.phis, self.thetas)

        self.ThetaIJ, self.chiIJ, self.pairs, self.chiRSS = Coff

        self.T_IJ = np.array([get_TspanIJ(spectra[ii],spectra[jj])
                              for ii,jj in zip(self.pairs[0],self.pairs[1])])
        
        self.spectra = spectra

    def SNR(self, Sh):
        """
        Calculate the signal-to-noise ratio of a given signal strain power
        spectral density, `Sh`. Must match frequency range and `df` of
        `self`.
        """
        integrand = Sh**2 / self.S_eff**2
        return np.sqrt(2.0 * self.Tspan * np.trapz(y=integrand,
                                                   x=self.freqs,
                                                   axis=0))

    @property
    def S_eff(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_S_eff'):
            ii = self.pairs[0]
            jj = self.pairs[1]
            kk = np.arange(len(self.chiIJ))
            num = self.T_IJ[kk] / self.Tspan * self.chiIJ[kk]**2
            series = num[:,np.newaxis] / (self.SnI[ii] * self.SnI[jj])
            self._S_eff = np.power(np.sum(series, axis=0),-0.5)
        return self._S_eff

    @property
    def S_effIJ(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_S_effIJ'):
            ii = self.pairs[0]
            jj = self.pairs[1]
            kk = np.arange(len(self.chiIJ))
            num = self.T_IJ[kk] / self.Tspan * self.chiIJ[kk]**2
            self._S_effIJ =  np.sqrt((self.SnI[ii] * self.SnI[jj])
                                     / num[:,np.newaxis])

        return self._S_effIJ
    
    def partial_hc(self):
        """_summary_
        """
        Tspan = get_Tspan(self.spectra)
        psr_idx = np.arange(self.phis.size)
        pairs = list(it.combinations(psr_idx,2))

        vals_gamma = 0
        vals_log10A = 0

        for I,J in pairs:
            Tspan_IJ = np.amax([self.spectra[I].toas.min(),self.spectra[J].toas.min()]) - np.amin([self.spectra[I].toas.max(),self.spectra[J].toas.max()])
            Chi_IJ = HD(thetas=[self.spectra[I].theta, self.spectra[J].theta], phis=[self.spectra[I].phi, self.spectra[J].phi])
            #computing hc_partial with respect to gamma
            vals_gamma = vals_gamma + Chi_IJ**2 * Tspan_IJ/Tspan * (
            self.spectra[I].partial_gamma_SI_Inv * 1/self.spectra[J].S_I + self.spectra[J].partial_gamma_SI_Inv * 1/self.spectra[I].S_I)

            #computing hc_partial with respect to log10A
            vals_log10A = vals_log10A + Chi_IJ**2 * Tspan_IJ/Tspan * (
            self.spectra[I].partial_log10A_SI_Inv * 1/self.spectra[J].S_I + self.spectra[J].partial_log10A_SI_Inv * 1/self.spectra[I].S_I)

        term_der_hc = -0.25 * self.freqs/self.h_c * self.S_eff**3

        der_hc_gamma =  term_der_hc * vals_gamma
        der_hc_log10A = term_der_hc * vals_log10A
        
        return der_hc_gamma, der_hc_log10A
    

    def hc_var(self, fim, log10A_gwb_mean, gamma_gwb_mean):
        return get_var_hc(self, fim, gamma_gwb_mean, log10A_gwb_mean)
        
        

class DeterSensitivityCurve(SensitivityCurve):
    '''
    Parameters
    ----------

    include_corr : bool
        Whether to include cross correlations from the GWB as an additional
        noise source in full PTA correlation matrix.
        (Has little to no effect and adds a lot of computation time.)

    A_GWB : float
        Value of GWB amplitude for use in cross correlations.
    '''
    def __init__(self, spectra, pulsar_term=True,
                 include_corr=False, A_GWB=None):
        super().__init__(spectra)
        self.T_I = np.array([sp.toas.max()-sp.toas.min() for sp in spectra])
        self.pulsar_term = pulsar_term
        self.include_corr = include_corr
        if include_corr:
            self.spectra = spectra
            if A_GWB is None:
                self.A_GWB = 1e-15
            else:
                self.A_GWB = A_GWB
            Coff = HellingsDownsCoeff(self.phis, self.thetas)
            self.ThetaIJ, self.chiIJ, self.pairs, self.chiRSS = Coff
            self.T_IJ = np.array([get_TspanIJ(spectra[ii],spectra[jj])
                                  for ii,jj in zip(self.pairs[0],
                                                   self.pairs[1])])
            self.NcalInvI = np.array([sp.NcalInv for sp in spectra])

    def SNR(self, h0):
        r'''
        Calculate the signal-to-noise ratio of a source given the strain
        amplitude. This is based on Equation (79) from Hazboun, et al., 2019
        `[1]`_.

        .. math::
            \rho(\hat{n})=h_0\sqrt{\frac{T_{\rm obs}}{S_{\rm eff}(f_0 ,\hat{k})}}

        .. _[1]: https://arxiv.org/abs/1907.04341
        '''
        return h0 * np.sqrt(self.Tspan / self.S_eff)

    @property
    def S_eff(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_S_eff'):
            t_I = self.T_I / self.Tspan
            elements = t_I[:,np.newaxis] / self.SnI
            sum1 = np.sum(elements, axis=0)
            if self.include_corr:
                sum = 0
                ii = self.pairs[0]
                jj = self.pairs[1]
                kk = np.arange(len(self.chiIJ))
                num = self.T_IJ[kk] / self.Tspan * self.chiIJ[kk]
                summand = num[:,np.newaxis] * self.NcalInvIJ
                summand *= resid_response(self.freqs)[np.newaxis,:]
                sum2 = np.sum(summand, axis=0)
            norm = 4./5 if self.pulsar_term else 2./5
            self._S_eff = np.power(norm * sum1,-1)
        return self._S_eff


def HD(phis,thetas):
    return HellingsDownsCoeff(np.array(phis),np.array(thetas))[1][0]


def HellingsDownsCoeff(phi, theta, autocorr=False):
    """
    Calculate Hellings and Downs coefficients from two lists of sky positions.

    Parameters
    ----------

    phi : array, list
        Pulsar axial coordinate.

    theta : array, list
        Pulsar azimuthal coordinate.

    Returns
    -------

    ThetaIJ : array
        An Npair-long array of angles between pairs of pulsars.

    chiIJ : array
        An Npair-long array of Hellings and Downs relation coefficients.

    pairs : array
        A 2xNpair array of pair indices corresponding to input order of sky
        coordinates.

    chiRSS : float
        Root-sum-squared value of all Hellings-Downs coefficients.

    """

    Npsrs = len(phi)
    # Npairs = np.int(Npsrs * (Npsrs-1) / 2.)
    psr_idx = np.arange(Npsrs)
    pairs = list(it.combinations(psr_idx,2))
    first, second = list(map(list, zip(*pairs)))
    cosThetaIJ = np.cos(theta[first]) * np.cos(theta[second]) \
                    + np.sin(theta[first]) * np.sin(theta[second]) \
                    * np.cos(phi[first] - phi[second])
    if autocorr:
        first.extend(psr_idx)
        second.extend(psr_idx)
        cosThetaIJ = np.append(cosThetaIJ,np.zeros(Npsrs))
    X = (1. - cosThetaIJ) / 2.
    chiIJ = [1.5*x*np.log(x) - 0.25*x + 0.5 if x!=0 else 1. for x in X]
    chiIJ = np.array(chiIJ)

    # calculate rss (root-sum-squared) of Hellings-Downs factor
    chiRSS = np.sqrt(np.sum(chiIJ**2))
    return np.arccos(cosThetaIJ), chiIJ, np.array([first,second]), chiRSS

def ScalarTensorCoeff(phi, theta, norm='std'):
    """
    Calculate Scalar-Tensor overlap reduction coefficients for alternative
    polarizations from two lists of sky positions.

    Parameters
    ----------

    phi : array, list
        Pulsar axial coordinate.

    theta : array, list
        Pulsar azimuthal coordinate.

    Returns
    -------

    ThetaIJ : array
        An Npair-long array of angles between pairs of pulsars.

    chiIJ : array
        An Npair-long array of Scalar Tensor ORF coefficients.

    pairs : array
        A 2xNpair array of pair indices corresponding to input order of sky
        coordinates.

    chiRSS : float
        Root-sum-squared value of all Scalar Tensor ORF coefficients.

    """

    Npsrs = len(phi)
    # Npairs = np.int(Npsrs * (Npsrs-1) / 2.)
    psr_idx = np.arange(Npsrs)
    pairs = list(it.combinations(psr_idx,2))
    first, second = list(map(list, zip(*pairs)))
    cosThetaIJ = np.cos(theta[first]) * np.cos(theta[second]) \
                    + np.sin(theta[first]) * np.sin(theta[second]) \
                    * np.cos(phi[first] - phi[second])
    X = 3/8+1/8*cosThetaIJ
    chiIJ = [x if x!=0 else 1. for x in X]
    chiIJ = np.array(chiIJ)

    # calculate rss (root-sum-squared) of Hellings-Downs factor
    chiRSS = np.sqrt(np.sum(chiIJ**2))
    return np.arccos(cosThetaIJ), chiIJ, np.array([first,second]), chiRSS

def DipoleCoeff(phi, theta, norm='std'):
    """
    Calculate Dipole overlap reduction coefficients from two lists of sky
    positions.

    Parameters
    ----------

    phi : array, list
        Pulsar axial coordinate.

    theta : array, list
        Pulsar azimuthal coordinate.

    Returns
    -------

    ThetaIJ : array
        An Npair-long array of angles between pairs of pulsars.

    chiIJ : array
        An Npair-long array of Dipole ORF coefficients.

    pairs : array
        A 2xNpair array of pair indices corresponding to input order of sky
        coordinates.

    chiRSS : float
        Root-sum-squared value of all Dipole ORF coefficients.

    """

    Npsrs = len(phi)
    # Npairs = np.int(Npsrs * (Npsrs-1) / 2.)
    psr_idx = np.arange(Npsrs)
    pairs = list(it.combinations(psr_idx,2))
    first, second = list(map(list, zip(*pairs)))
    cosThetaIJ = np.cos(theta[first]) * np.cos(theta[second]) \
                    + np.sin(theta[first]) * np.sin(theta[second]) \
                    * np.cos(phi[first] - phi[second])
    X = 0.5*cosThetaIJ
    chiIJ = [x if x!=0 else 1. for x in X]
    chiIJ = np.array(chiIJ)

    # calculate rss (root-sum-squared) of Hellings-Downs factor
    chiRSS = np.sqrt(np.sum(chiIJ**2))
    return np.arccos(cosThetaIJ), chiIJ, np.array([first,second]), chiRSS

def MonopoleCoeff(phi, theta, norm='std'):
    """
    Calculate Monopole overlap reduction coefficients from two lists of sky
    positions.

    Parameters
    ----------

    phi : array, list
        Pulsar axial coordinate.

    theta : array, list
        Pulsar azimuthal coordinate.

    Returns
    -------

    ThetaIJ : array
        An Npair-long array of angles between pairs of pulsars.

    chiIJ : array
        An Npair-long array of Dipole ORF coefficients.

    pairs : array
        A 2xNpair array of pair indices corresponding to input order of sky
        coordinates.

    chiRSS : float
        Root-sum-squared value of all Dipole ORF coefficients.

    """

    Npsrs = len(phi)
    # Npairs = np.int(Npsrs * (Npsrs-1) / 2.)
    psr_idx = np.arange(Npsrs)
    pairs = list(it.combinations(psr_idx,2))
    first, second = list(map(list, zip(*pairs)))
    cosThetaIJ = np.cos(theta[first]) * np.cos(theta[second]) \
                    + np.sin(theta[first]) * np.sin(theta[second]) \
                    * np.cos(phi[first] - phi[second])
    chiIJ = np.ones_like(cosThetaIJ)

    # calculate rss (root-sum-squared) of Hellings-Downs factor
    chiRSS = np.sqrt(np.sum(chiIJ**2))
    return np.arccos(cosThetaIJ), chiIJ, np.array([first,second]), chiRSS


def get_Tspan(psrs):
    """
    Returns the total timespan from a list or arry of Pulsar objects, psrs.
    """
    try:
        last = np.amax([p.toas.max() for p in psrs])
        first = np.amin([p.toas.min() for p in psrs])
        tspan = last-first
    except ValueError:
        tspan = 0
    return tspan

def get_TspanIJ(psr1,psr2):
    """
    Returns the overlapping timespan of two Pulsar objects, psr1/psr2.
    """
    start = np.amax([psr1.toas.min(),psr2.toas.min()])
    stop = np.amin([psr1.toas.max(),psr2.toas.max()])
    return stop - start

def corr_from_psd(freqs, psd, toas, fast=True):
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
        tm = np.sqrt(psd*df)*np.exp(1j*2*np.pi*freqs*toas[:,np.newaxis])
        integrand = jnp.matmul(tm, np.conjugate(tm.T))
        return np.real(integrand)
    else: #Makes much larger arrays, but uses np.trapz
        t1, t2 = np.meshgrid(toas, toas, indexing='ij')
        tm = np.abs(t1-t2)
        integrand = psd*np.cos(2*np.pi*freqs*tm[:,:,np.newaxis])#df*
        return jnp.trapezoid(integrand, axis=2, x=freqs)#np.sum(integrand,axis=2)#

def corr_from_psdIJ(freqs, psd, toasI, toasJ, fast=True):
    """
    Calculates the correlation matrix over a set of TOAs for a given power
    spectral density for two pulsars.

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
        tmI = np.sqrt(psd*df)*np.exp(1j*2*np.pi*freqs*toasI[:,np.newaxis])
        tmJ = np.sqrt(psd*df)*np.exp(1j*2*np.pi*freqs*toasJ[:,np.newaxis])
        integrand = jnp.matmul(tmI, np.conjugate(tmJ.T))
        return np.real(integrand)
    else: #Makes much larger arrays, but uses np.trapz
        t1, t2 = np.meshgrid(toasI, toasJ, indexing='ij')
        tm = np.abs(t1-t2)
        integrand = psd*np.cos(2*np.pi*freqs*tm[:,:,np.newaxis])#df*
        return np.trapz(integrand, axis=2, x=freqs)

def quantize_fast(toas, toaerrs, flags=None, dt=0.1):
    r"""
    Function to quantize and average TOAs by observation epoch. Used especially
    for NANOGrav multiband data.

    Pulled from `[3]`_.

    .. _[3]: https://github.com/vallis/libstempo/blob/master/libstempo/toasim.py

    Parameters
    ----------

    times : array
        TOAs for a pulsar.

    flags : array, optional
        Flags for TOAs.

    dt : float
        Coarse graining time [days].
    """
    isort = np.argsort(toas)

    bucket_ref = [toas[isort[0]]]
    bucket_ind = [[isort[0]]]
    dt *= (24*3600)
    for i in isort[1:]:
        if toas[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(toas[i])
            bucket_ind.append([i])

    avetoas = np.array([np.mean(toas[l]) for l in bucket_ind],'d')
    avetoaerrs = np.array([sps.hmean(toaerrs[l]) for l in bucket_ind],'d')
    if flags is not None:
        aveflags = np.array([flags[l[0]] for l in bucket_ind])

    U = np.zeros((len(toas),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1

    if flags is not None:
        return avetoas, avetoaerrs, aveflags, U, bucket_ind
    else:
        return avetoas, avetoaerrs, U, bucket_ind

def SimCurve():
    raise NotImplementedError()

def red_noise_powerlaw(A, freqs, gamma=None, alpha=None):
    r"""
    Add power law red noise to the prefit residual power spectral density.
    As :math:`P=A^2(f/fyr)^{-\gamma}`

    Parameters
    ----------
    A : float
        Amplitude of red noise.

    gamma : float
        Spectral index of red noise powerlaw.

    freqs : array
        Frequencies at which to calculate the red noise power law.
    """
    if gamma is None and alpha is not None:
        gamma = 3-2*alpha
    elif ((gamma is None and alpha is None)
          or (gamma is not None and alpha is not None)):
        ValueError('Must specify one version of spectral index.')

    return A**2*(freqs/fyr)**(-gamma)/(12*np.pi**2) * yr_sec**3

def psd_from_background_realization(background_hc, freqs):
    r"""
    Calculate the power spectral density with given background strain and frequency binning.

    Parameters
    ----------
    background_hc : array
        Characteristic strain (hc) of background at each frequency.

    freqs : array
        Frequency bins over which the background is stored.

    Returns
    -------
    S_h : array
        the power spectral density of the background
    """

    return background_hc**2 / (12 * np.pi**2 * freqs[:,np.newaxis]**3)

def S_h(A, alpha, freqs):
    r"""
    Add power law red noise to the prefit residual power spectral density.
    As S_h=A^2*(f/fyr)^(2*alpha)/f

    Parameters
    ----------
    A : float
        Amplitude of red noise.

    alpha : float
        Spectral index of red noise powerlaw.

    freqs : array
        Array of frequencies at which to calculate S_h.
    """

    return A**2*(freqs/fyr)**(2*alpha) / freqs

def Agwb_from_Seff_plaw(freqs, Tspan, SNR, S_eff, gamma=13/3., alpha=None):
    r"""
    Must supply numpy.ndarrays.
    """
    if alpha is None:
        alpha = (3-gamma)/2
    else:
        pass

    if hasattr(alpha,'size'):
        fS_sqr = freqs**2 * S_eff**2
        integrand = (freqs[:,np.newaxis]/fyr)**(4*alpha)
        integrand /= fS_sqr[:,np.newaxis]
        fintegral = np.trapz(integrand, x=freqs,axis=0)
    else:
        integrand = (freqs/fyr)**(4*alpha) / freqs**2 / S_eff**2
        fintegral = np.trapz(integrand, x=freqs)

    return np.sqrt(SNR)/np.power(2 * Tspan * fintegral, 1/4.)

def PI_hc(freqs, Tspan, SNR, S_eff, N=200):
    '''Power law-integrated characteristic strain.'''
    alpha = np.linspace(-1.75, 1.25, N)
    h = Agwb_from_Seff_plaw(freqs=freqs, Tspan=Tspan, SNR=SNR,
                            S_eff=S_eff, alpha=alpha)
    plaw = np.dot((freqs[:,np.newaxis]/fyr)**alpha,h[:,np.newaxis]*np.eye(N))
    PI_sensitivity = np.amax(plaw, axis=1)

    return PI_sensitivity, plaw

def get_dt(toas):
    '''Returns average dt between observation epochs given toas.'''
    toas = make_quant(toas, u.s)
    return np.diff(np.unique(np.round(toas.to('day')))).mean()

def make_quant(param, default_unit):
    """Convenience function to intialize a parameter as an astropy quantity.
    param == parameter to initialize.
    default_unit == string that matches an astropy unit, set as
                    default for this parameter.

    returns:
        an astropy quantity

    example:
        self.f0 = make_quant(f0,'MHz')
    """
    default_unit = u.core.Unit(default_unit)
    if hasattr(param, 'unit'):
        try:
            param.to(default_unit)
        except u.UnitConversionError:
            raise ValueError("Quantity {0} with incompatible unit {1}"
                             .format(param, default_unit))
        quantity = param.to(default_unit)
    else:
        quantity = param * default_unit

    return quantity


def spectra_h5_creation(dir:str, psr_names:list, gamma_gwb, A_gwb, freqs):
    """converts h5 file of spectra objects back to list of spectrum objects
    """
    class h5_spectra:
        def __init__(self, freqs, phi, theta, toas, pdist, toaerrs, N, G, S_I, designmatrix, partial_gamma_SI_Inv, partial_log10A_SI_Inv, name):
            self.phi = phi
            self.theta = theta
            self.toas = toas
            self.toaerrs = toaerrs
            self.freqs = freqs
            self.S_I = S_I
            self.N = N
            self.G = G
            self.pdist = pdist
            self.partial_gamma_SI_Inv = partial_gamma_SI_Inv
            self.partial_log10A_SI_Inv = partial_log10A_SI_Inv
            self.designmatrix = designmatrix
            self.name = name
            

    spectra = []
            
    with h5py.File(dir, 'r') as f:
        for name in psr_names:
            psr_h5 = f[name]
            psr = h5_spectra(freqs=freqs, theta=psr_h5['theta'][:][0], phi=psr_h5['phi'][:][0], S_I=psr_h5['S_I'][:], partial_gamma_SI_Inv=psr_h5['partial_gamma_SI_Inv'][:],
                             partial_log10A_SI_Inv=psr_h5['partial_log10A_SI_Inv'][:], toas=psr_h5['toas'][:], toaerrs=psr_h5['toaerrs'][:], G=psr_h5['G'][:],
                             N = psr_h5['N'][:], designmatrix=psr_h5['Mmat'][:], pdist=psr_h5['pdist'][:],  name=name)
            spectra_psr = Spectrum(psr, freqs=psr.freqs, gamma_gwb=gamma_gwb, A_gwb=A_gwb)
            spectra.append(spectra_psr)
    return spectra


################## Pre-Made Sensitivity Curves#############
def nanograv_11yr_deter():
    '''
    Returns a `DeterSensitivityCurve` object built using with the NANOGrav
    11-year data set.
    '''
    path = sc_dir + 'nanograv_11yr_deter.sc'
    with open(path, "rb") as fin:
        sc = pickle.load(fin)
        sc.filepath = path
    return sc

def nanograv_11yr_stoch():
    '''
    Returns a `GWBSensitivityCurve` object built using with the NANOGrav 11-year
    data set.
    '''
    path = sc_dir + 'nanograv_11yr_stoch.sc'
    with open(path, "rb") as fin:
        sc = pickle.load(fin)
        sc.filepath = path
    return sc