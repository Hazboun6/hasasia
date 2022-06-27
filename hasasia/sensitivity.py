# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np
import itertools as it
import scipy.stats as sps
import scipy.linalg as sl
import os, pickle
from astropy import units as u

import hasasia
from .utils import create_design_matrix

current_path = os.path.abspath(hasasia.__path__[0])
sc_dir = os.path.join(current_path,'sensitivity_curves/')

__all__ =['GWBSensitivityCurve',
          'DeterSensitivityCurve',
          'Pulsar',
          'Spectrum',
          'R_matrix',
          'G_matrix',
          'get_Tf',
          'get_NcalInv',
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
    L = sl.cholesky(N)
    Linv = sl.inv(L)
    U,s,_ = sl.svd(np.matmul(Linv,M), full_matrices=True)
    Id = np.eye(M.shape[0])
    S = np.zeros_like(M)
    S[:m,:m] = np.diag(s)
    inner = sl.inv(np.matmul(S.T,S))
    outer = np.matmul(S,np.matmul(inner,S.T))

    return Id - np.matmul(L,np.matmul(np.matmul(U,outer),np.matmul(U.T,Linv)))

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
    U, _ , _ = sl.svd(M, full_matrices=True)

    return U[:,m:]

def get_Tf(designmatrix, toas, N=None, nf=200, fmin=None, fmax=2e-7,
           freqs=None, exact_astro_freqs = False,
           from_G=True, twofreqs=False, Gmatrix=None):
    """
    Calculate the transmission function for a given pulsar design matrix, TOAs
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
        ff = np.logspace(np.log10(fmin), np.log10(fmax), nf,dtype='float128')
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
        Gtilde = np.zeros((ff.size,G.shape[1]),dtype='complex128')
        Gtilde = np.dot(np.exp(1j*2*np.pi*ff[:,np.newaxis]*toas),G)
        Tmat = np.matmul(np.conjugate(Gtilde),Gtilde.T)/N_TOA
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

    Gtilde = np.zeros((ff.size,G.shape[1]),dtype='complex128')
    #N_freqs x N_TOA-N_par

    # Note we do not include factors of NTOA or Timespan as they cancel
    # with the definition of Ncal
    Gtilde = np.dot(np.exp(1j*2*np.pi*ff[:,np.newaxis]*toas),G)
    # N_freq x N_TOA-N_par

    Ncal = np.matmul(G.T,np.matmul(psr.N,G)) #N_TOA-N_par x N_TOA-N_par
    NcalInv = sl.inv(Ncal) #N_TOA-N_par x N_TOA-N_par

    TfN = np.matmul(np.conjugate(Gtilde),np.matmul(NcalInv,Gtilde.T)) / 2
    if return_Gtilde_Ncal:
        return np.real(TfN), Gtilde, Ncal
    elif full_matrix:
        return np.real(TfN)
    else:
        return np.real(np.diag(TfN)) / get_Tspan([psr])

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

    designmatrix : array
        Design matrix for pulsar's timing model. N_TOA x N_param.

    N : array
        Covariance matrix for the pulsar. N_TOA x N_TOA. Made from toaerrs
        if not provided.

    pdist : astropy.quantity, float
        Earth-pulsar distance. Default units is kpc.

    """
    def __init__(self, toas, toaerrs, phi=None, theta=None,
                 designmatrix=None, N=None, pdist=1.0*u.kpc):
        self.toas = toas
        self.toaerrs = toaerrs
        self.phi = phi
        self.theta = theta
        self.pdist = make_quant(pdist,'kpc')

        if N is None:
            self.N = np.diag(toaerrs**2) #N ==> weights
        else:
            self.N = N

        if designmatrix is None:
            self.designmatrix = create_design_matrix(toas, RADEC=True,
                                                     PROPER=True, PX=True)
        else:
            self.designmatrix = designmatrix

    @property
    def G(self):
        """Inverse Noise Weighted Transmission Function."""
        if not hasattr(self, '_G'):
            self._G = G_matrix(designmatrix=self.designmatrix)
        return self._G

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
    def __init__(self, psr, nf=400, fmin=None, fmax=2e-7,
                 freqs=None, tm_fit=True, **Tf_kwargs):
        self._H_0 = 72 * u.km / u.s / u.Mpc
        self.toas = psr.toas
        self.toaerrs = psr.toaerrs
        self.phi = psr.phi
        self.theta = psr.theta
        self.N = psr.N
        self.G = psr.G
        self.designmatrix = psr.designmatrix
        self.pdist = psr.pdist
        self.tm_fit = tm_fit
        self.Tf_kwargs = Tf_kwargs
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
    def Omega_gw(self):
        """Energy Density sensitivity"""
        self._Omega_gw = ((2*np.pi**2/3) * self.freqs**3 * self.S_eff
                           / self._H_0.to('Hz').value**2)
        return self._Omega_gw

    @property
    def H_0(self):
        """Hubble Constant. Assumed to be in units of km /(s Mpc) unless
        supplied as an `astropy.quantity`. """
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

    @property
    def NcalInvIJ(self):
        """
        Inverse Noise Weighted Transmission Function that includes
        cross-correlation noise from GWB.
        """
        if not hasattr(self,'_NcalInvIJ'):
            self._NcalInvIJ = get_NcalInvIJ(psrs=self.spectra,
                                            A_GWB=self.A_GWB,
                                            freqs=self.freqs,
                                            full_matrix=True)

        return self._NcalInvIJ


def HD(phis,thetas):
    return HellingsDownsCoeff(np.array(phis),np.array(thetas))[1][0]


def get_NcalInvIJ(psrs, A_GWB, freqs, full_matrix=False,
                  return_Gtilde_Ncal=False):
    r"""
    Calculate the inverse-noise-wieghted transmission function for a given
    pulsar. This calculates
    :math:`\mathcal{N}^{-1}(f,f') , \; \mathcal{N}^{-1}(f)`
    in `[1]`_, see Equations (19-20).

    .. _[1]: https://arxiv.org/abs/1907.04341

    Parameters
    ----------

    psrs : list of hasasia.Pulsar objects
        List of hasasia.Pulsar objects to build NcalInvIJ


    Returns
    -------

    inverse-noise-weighted transmission function across two pulsars.

    """
    Npsrs = len(psrs)
    toas = np.concatenate([p.toas for p in psrs], axis=None)
    # make filter
    ff = np.tile(freqs, Npsrs)
    ## CHANGE BACK
    # G = sl.block_diag(*[G_matrix(p.designmatrix) for p in psrs])
    G = sl.block_diag(*[np.eye(p.toas.size) for p in psrs])
    Gtilde = np.zeros((ff.size, G.shape[1]), dtype='complex128')
    #N_freqs x N_TOA-N_par

    Gtilde = np.dot(np.exp(1j*2*np.pi*ff[:,np.newaxis]*toas),G)
    # N_freq x N_TOA-N_par
    #CHANGE BACK
    # psd = red_noise_powerlaw(A=A_GWB, gamma=13./3, freqs=freqs)
    psd = 2*(365.25*24*3600/40)*(1e-6)**2
    Ch_blocks = [[(HD([pc.phi,pr.phi],[pc.theta,pr.theta])
                   *corr_from_psdIJ(freqs=freqs, psd=psd, toasI=pc.toas,
                                    toasJ=pr.toas, fast=True))
                  if r!=c
                  else corr_from_psdIJ(freqs=freqs, psd=psd, toasI=pc.toas,
                                       toasJ=pr.toas, fast=True)
                  for r, pr in enumerate(psrs)]
                  for c, pc in enumerate(psrs)]

    C_h = np.block(Ch_blocks)

    C_n = sl.block_diag(*[p.N for p in psrs])
    # C_h = sl.block_diag(*[corr_from_psd(freqs=freqs, psd=psd,
    #                                     toas=p.toas, fast=True) for p in psrs])
    C = C_n + C_h
    Ncal = np.matmul(G.T, np.matmul(C, G)) #N_TOA-N_par x N_TOA-N_par
    NcalInv = sl.inv(Ncal) #N_TOA-N_par x N_TOA-N_par

    TfN = NcalInv#np.matmul(G, np.matmul(NcalInv, G.T))
    #np.matmul(np.conjugate(Gtilde),np.matmul(NcalInv,Gtilde.T)) / 2

    if return_Gtilde_Ncal:
        return np.real(TfN), Gtilde, Ncal
    elif full_matrix:
        return np.real(TfN), toas, ChiIJ
    else:
        return np.real(np.diag(TfN)) / get_Tspan(psrs)


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
    last = np.amax([p.toas.max() for p in psrs])
    first = np.amin([p.toas.min() for p in psrs])
    return last - first

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
        integrand = np.matmul(tm, np.conjugate(tm.T))
        return np.real(integrand)
    else: #Makes much larger arrays, but uses np.trapz
        t1, t2 = np.meshgrid(toas, toas, indexing='ij')
        tm = np.abs(t1-t2)
        integrand = psd*np.cos(2*np.pi*freqs*tm[:,:,np.newaxis])#df*
        return np.trapz(integrand, axis=2, x=freqs)#np.sum(integrand,axis=2)#

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
        integrand = np.matmul(tmI, np.conjugate(tmJ.T))
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
