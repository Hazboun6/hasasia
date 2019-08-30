# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np
import itertools as it
import scipy.stats as sps
from astropy import units as u

from .utils import create_design_matrix

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
    U,s,_ = np.linalg.svd(np.matmul(Linv,M), full_matrices=True)
    Id = np.eye(M.shape[0])
    S = np.zeros_like(M)
    S[:m,:m] = np.diag(s)
    inner = np.linalg.inv(np.matmul(S.T,S))
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
    U, _ , _ = np.linalg.svd(M, full_matrices=True)

    return U[:,m:]

def get_Tf(designmatrix, toas, N=None, nf=200, fmin=None, fmax=2e-7,
           freqs=None, exact_astro_freqs = False, from_G=True, twofreqs=False):
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
        G = G_matrix(M)
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
               return_Gtilde_Ncal=False):
    """
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

    G = G_matrix(psr.designmatrix)
    Gtilde = np.zeros((ff.size,G.shape[1]),dtype='complex128')
    #N_freqs x N_TOA-N_par

    NTOA = psr.toas
    # Note we do not include factors of NTOA or Timespan as they cancel
    # with the definition of Ncal
    Gtilde = np.dot(np.exp(1j*2*np.pi*ff[:,np.newaxis]*toas),G)
    # N_freq x N_TOA-N_par

    Ncal = np.matmul(G.T,np.matmul(psr.N,G)) #N_TOA-N_par x N_TOA-N_par
    NcalInv = np.linalg.inv(Ncal) #N_TOA-N_par x N_TOA-N_par

    TfN = np.matmul(np.conjugate(Gtilde),np.matmul(NcalInv,Gtilde.T)) / 2
    if return_Gtilde_Ncal:
        return np.real(TfN), Gtilde, Ncal
    elif full_matrix:
        return np.real(TfN)
    else:
        return np.real(np.diag(TfN)) / get_Tspan([psr])

def resid_response(freqs):
    """
    Returns the timing residual response function for a pulsar across as set of
    frequencies. See Equation (53) in `[1]`_.

    .. math::
        \\mathcal{R}(f)=\\frac{1}{12\pi^2\;f^2}

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

    """
    def __init__(self, toas, toaerrs, phi=None, theta=None,
                 designmatrix=None, N=None):
        self.toas = toas
        self.toaerrs = toaerrs
        self.phi = phi
        self.theta = theta

        if N is None:
            self.N = np.diag(toaerrs**2) #N ==> weights
        else:
            self.N = N

        if designmatrix is None:
            self.designmatrix = create_design_matrix(toas, RADEC=True,
                                                     PROPER=True, PX=True)
        else:
            self.designmatrix = designmatrix

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
                 freqs=None, **Tf_kwargs):
        self.toas = psr.toas
        self.toaerrs = psr.toaerrs
        self.phi = psr.phi
        self.theta = psr.theta
        self.N = psr.N
        self.designmatrix = psr.designmatrix
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
                                       freqs=self.freqs, from_G=True,
                                       **self.Tf_kwargs)
        return self._Tf

    @property
    def NcalInv(self):
        """Inverse Noise Weighted Transmission Function."""
        if not hasattr(self, '_NcalInv'):
            self._NcalInv = get_NcalInv(psr=self,freqs=self.freqs)
        return self._NcalInv

    @property
    def S_I(self):
        """Strain power sensitivity for this pulsar. Equation (74) in `[1]`_

        .. math::
            S_I=\\frac{1}{\mathcal{N}^{-1}\;\mathcal{R}}

        .. _[1]: https://arxiv.org/abs/1907.04341
        """
        if not hasattr(self, '_S_I'):
            self._S_I = 1/resid_response(self.freqs)/self.NcalInv
        return self._S_I

    @property
    def S_R(self):
        """Residual power sensitivity for this pulsar.

        .. math::
            S_R=\\frac{1}{\mathcal{N}^{-1}}

        """
        if not hasattr(self, '_S_R'):
            self._S_R = 1/self.NcalInv
        return self._S_R

    @property
    def h_c(self):
        """Characteristic strain sensitivity for this pulsar.

        .. math::
            h_c=\\sqrt{f\;S_I}
        """
        if not hasattr(self, '_h_c'):
            self._h_c = np.sqrt(self.freqs * self.S_I)
        return self._h_c

    @property
    def Omega_gw(self):
        """Energy Density sensitivity.

        .. math::
            \\Omega_{gw}=\\frac{2\\pi^2}{3\;H_0^2}f^3\;S_I
        """
        self._Omega_gw = ((2*np.pi**2/3) * self.freqs**3 * self.S_I
                           / self._H_0.to('Hz').value**2)
        return self._Omega_gw

    def add_white_noise_power(self, sigma=None, dt=None, vals=False):
        """
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
        """
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
        """Add any spectrum of noise. Must match length of frequency array.

        **Note:** All noise information is furnished by the covariance matrix in
        the `hasasia.Pulsar` object, this is simply useful for bookkeeping and
        plots.
        """
        self._psd_prefit += noise


class SensitivityCurve(object):
    """
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
        """Hubble Constant. Must be given in """
        self._H_0 = make_quant(self._H_0,'km /(s Mpc)')
        return self._H_0


class GWBSensitivityCurve(SensitivityCurve):
    """
    Class to produce a sensitivity curve for a gravitational wave
    background, using Hellings-Downs spatial correlations.
    """
    def __init__(self, spectra):
        super().__init__(spectra)
        HDCoff = HellingsDownsCoeff(self.phis, self.thetas)
        self.ThetaIJ, self.chiIJ, self.pairs, self.chiRSS = HDCoff

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
    def __init__(self, spectra):
        super().__init__(spectra)
        self.T_I = np.array([sp.toas.max()-sp.toas.min() for sp in spectra])

    @property
    def S_eff(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_S_eff'):
            t_I = self.T_I / self.Tspan
            series = t_I[:,np.newaxis] / self.SnI
            self._S_eff = np.power((4./5.) * np.sum(series, axis=0),-1)
        return self._S_eff


def HellingsDownsCoeff(phi, theta):
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
    X = (1. - cosThetaIJ) / 2.
    chiIJ = [1.5*x*np.log(x) - 0.25*x + 0.5 if x!=0 else 1. for x in X]
    chiIJ = np.array(chiIJ)

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
        t1, t2 = np.meshgrid(toas, toas)
        tm = np.abs(t1-t2)
        integrand = psd*np.cos(2*np.pi*freqs*tm[:,:,np.newaxis])#df*
        return np.trapz(integrand, axis=2, x=freqs)#np.sum(integrand,axis=2)#

def quantize_fast(toas, toaerrs, flags=None, dt=0.1):
    """
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
        return avetoas, avetoaerrs, aveflags, U
    else:
        return avetoas, avetoaerrs, U

def SimCurve():
    raise NotImplementedError()

def red_noise_powerlaw(A, freqs, gamma=None, alpha=None):
    """
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
    """
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
    """
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
    return np.round(np.diff(np.unique(np.round(toas.to('day')))).mean())

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
