# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np
import itertools as it
import scipy.stats as sps
import mpmath
from astropy import units as u

from . import sim

__all__ =['R_matrix',
          'G_matrix',
          'get_Tf',
          'get_tmm_noise',
          'resid_response',
          'Pulsar',
          'Spectrum',
          'GWBSensitivityCurve',
          'DeterSensitivityCurve',
          'SensitivityCurve',
          'HellingsDownsCoeff',
          'get_Tspan',
          'get_TspanIJ',
          'corr_from_psd',
          'rn_corr_from_psd',
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

    Return
    ------
    R matrix

    """
#     w = 1/np.sqrt(np.diagonal(N))
#     W = np.diag(w)#np.sqrt(1/N)
#     # w = np.diagonal(W)
#
#     u, s, v = np.linalg.svd((w * designmatrix.T).T,full_matrices=False)
# #     print(u.shape,s.shape,v.shape)
#     return np.eye(N.shape[0]) - (1/w * np.dot(u, np.dot(u.T, W)).T).T
    M = designmatrix
    n,m = M.shape

    # if N.ndim ==1:
    #     Ninv = np.diag(1/N)
    # else:
    #     Ninv = np.linalg.inv(N)

    L = np.linalg.cholesky(N)
    Linv = np.linalg.inv(L)
    U,s,_ = np.linalg.svd(np.matmul(Linv,M), full_matrices=True)
    Id = np.eye(M.shape[0])
    S = np.zeros_like(M)
    S[:m,:m] = np.diag(s)
    inner = np.linalg.inv(np.matmul(S.T,S))
    outer = np.matmul(S,np.matmul(inner,S.T))

    return Id - np.matmul(L,np.matmul(np.matmul(U,outer),np.matmul(U.T,Linv)))

    # MtNinvM = np.matmul(M.T,np.matmul(Ninv,M))
    # try:
    #     MtNinvMinv = np.linalg.inv(MtWsqM)
    # except:
    #     L = np.linalg.cholesky(MtNinvM)
    #     MtNinvMinv =np.matmul(np.linalg.inv(L.T),np.linalg.inv(L))
    #
    # Id = np.eye(M.shape[0])
    #
    # return Id - np.matmul(np.matmul(M,np.matmul(MtNinvMinv,M.T)),Ninv)

def G_matrix(designmatrix):
    """
    Create G matrix as defined in van Haasteren 2013

    Parameters
    ----------

    designmatrix : array
        Design matrix for a pulsar timing model.

    Return
    ------
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

    fmax: float, optional
        Maximum frequency at which to calculate transmission function.

    exact_astro_freqs: bool, optional
        Whether to use exact 1/year and 2/year frequency values in calculation.

    from_G: bool, optional
        Whether to use G matrix for transmission function calculate. Default is
        False, in which case R matrix is used.
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

def get_tmm_noise(psr, nf=200, fmin=None, fmax=2e-7, freqs=None,
                  exact_yr_freqs = False, full_matrix=False,
                  return_Gtilde_Ncal=False):
    """
    Calculate the timing model marginalized noise power spectral density for a
    given pulsar.

    Parameters
    ----------

    psr : array
        Pulsar object.

    nf : int, optional
        Number of frequencies at which to calculate transmission function.

    fmin : float, optional
        Minimum frequency at which to calculate transmission function.

    fmax: float, optional
        Maximum frequency at which to calculate transmission function.

    exact_yr_freqs: bool, optional
        Whether to use exact 1/year and 2/year frequency values in calculation.

    Returns
    -------

    noise weighted transmission function

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
    """Timing residual response function."""
    return 1/(12 * np.pi**2 * freqs**2)

class Pulsar(object):
    """
    Class to encode information about individual pulsars.
    """
    def __init__(self, toas, toaerrs, phi=None, theta=None,
                 designmatrix=None, N=None):
        """
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
        self.toas = toas
        self.toaerrs = toaerrs
        self.phi = phi
        self.theta = theta

        if N is None:
            self.N = np.diag(toaerrs**2) #N ==> weights
        else:
            self.N = N

        if designmatrix is None:
            self.designmatrix = sim.create_design_matrix(toas, RADEC=True,
                                                         PROPER=True, PX=True)
        else:
            self.designmatrix = designmatrix

class Spectrum(object):
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
            self._psd_postfit = self.psd_prefit * self.Tf
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
    def Tf_simp(self):
        if not hasattr(self, '_Tf_simp'):
            self._Tf_simp,_,_ = get_Tf(designmatrix=self.designmatrix,
                                  toas=self.toas, N=self.N,
                                  freqs=self.freqs,from_G=True,**self.Tf_kwargs)
        return self._Tf_simp

    @property
    def Tf(self):
        if not hasattr(self, '_Tf'):
            self._Tf = get_tmm_noise(psr=self,freqs=self.freqs)
        return self._Tf

    @property
    def Sn(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_Sn'):
            self._Sn = 1/resid_response(self.freqs)/self.Tf
        return self._Sn

    @property
    def S_R(self):
        """Residual power sensitivity. """
        if not hasattr(self, '_S_R'):
            self._S_R = 1/self.Tf
        return self._S_R

    @property
    def h_c(self):
        """Characteristic strain sensitivity"""
        if not hasattr(self, '_h_c'):
            self._h_c = np.sqrt(self.freqs * self.Sn)
        return self._h_c

    @property
    def Omega_gw(self):
        """Energy Density sensitivity"""
        raise NotImplementedError()

    def add_white_noise_power(self, sigma=None, dt=None, vals=False):
        """
        Add power law red noise to the prefit residual power spectral density.

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
        As P=A^2*(f/fyr)^-gamma*df

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
        # df = np.diff(ff)
        # df = np.append(df[0],df)
        red_noise = A**2*(ff/fyr)**(-gamma)/(12*np.pi**2) * yr_sec**3# * df
        self._psd_prefit += red_noise
        if vals:
            return red_noise

    def add_noise_power(self,noise):
        """Add any spectrum of noise."""
        self._psd_prefit += noise


class SensitivityCurve(object):
    """
    Class for constructing PTA sensitivity curves.
    """
    def __init__(self, spectra):
        if not isinstance(spectra, list):
            raise ValueError('Must provide list of spectra!!')

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

        self.SnI = np.array([sp.Sn for sp in spectra])

    @property
    def S_eff(self):
        """Strain power sensitivity. """
        raise NotImplementedError('Effective Strain Power Sensitivity method must be defined.')

    @property
    def h_c(self):
        """Characteristic strain sensitivity"""
        if not hasattr(self, '_h_c'):
            self._h_c = np.sqrt(self.freqs * self.S_eff)
        return self._h_c

    @property
    def Omega_gw(self):
        """Energy Density sensitivity"""
        raise NotImplementedError()


class GWBSensitivityCurve(SensitivityCurve):
    def __init__(self, spectra):
        super().__init__(spectra)
        HDCoff = HellingsDownsCoeff(self.phis, self.thetas)
        self.ThetaIJ, self.alphaIJ, self.pairs, self.alphaRSS = HDCoff

        self.T_IJ = np.array([get_TspanIJ(spectra[ii],spectra[jj])
                              for ii,jj in zip(self.pairs[0],self.pairs[1])])

    @property
    def S_eff(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_S_eff'):
            ii = self.pairs[0]
            jj = self.pairs[1]
            kk = np.arange(len(self.alphaIJ))
            num = self.T_IJ[kk] / self.Tspan * self.alphaIJ[kk]**2
            series = num[:,np.newaxis] / (self.SnI[ii] * self.SnI[jj])
            self._S_eff = np.power(np.sum(series, axis=0),-0.5)
        return self._S_eff

    @property
    def S_effIJ(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_S_effIJ'):
            ii = self.pairs[0]
            jj = self.pairs[1]
            kk = np.arange(len(self.alphaIJ))
            num = self.T_IJ[kk] / self.Tspan * self.alphaIJ[kk]**2
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

    theta: array, list
        Pulsar azimuthal coordinate.

    Returns
    -------

    ThetaIJ : array
        An Npair-long array of angles between pairs of pulsars.

    alphaIJ : array
        An Npair-long array of Hellings and Downs relation coefficients.

    pairs: array
        A 2xNpair array of pair indices corresponding to input order of sky
        coordinates.

    alphaRSS : float
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
    alphaIJ = [1.5*x*np.log(x) - 0.25*x/4. + 0.5 if x!=0 else 1. for x in X]
    alphaIJ = np.array(alphaIJ)

    # calculate rss (root-sum-squared) of Hellings-Downs factor
    alphaRSS = np.sqrt(np.sum(alphaIJ**2))
    return np.arccos(cosThetaIJ), alphaIJ, np.array([first,second]), alphaRSS

def get_Tspan(psrs):
    """
    Returns the total timespan from a list of Pulsar objects, psrs.
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

def rn_corr_from_psd(toas, fL, Amp, alpha=1/2):
    #Calculate taus
    t1, t2 = np.meshgrid(toas, toas)
    tau = np.abs(t1-t2).flatten()

    #Ignore zero values along diagonal
    hyp1F2 = [mpmath.hyp1f2(alpha-1,1/2.,alpha,-(np.pi*fL*t)**2)
              if t!=0 else 0 for t in tau] #Calculate all nonzero values
    hyp1F2 = np.array(hyp1F2).astype('float')
    tau[tau==0]=1
    hyp1F2 *= ((fL*tau)**(2*(alpha-1)))/(2*(1-alpha))
    hyp1F2.resize(toas.size, toas.size)
    tau.resize(toas.size, toas.size)
    second_term = (2*np.pi)**(2*(alpha-1)) * np.cos(np.pi*alpha)
    try:
        second_term *= float(mpmath.gamma(2*(alpha-1)))
    except ValueError:
        second_term *= 1e40#mpmath.gamma(2*(alpha-1))
    coeff = tau**2 * (fyr*tau)**(-2*alpha) / 12 / np.pi**2
    corr = Amp**2 * coeff*(hyp1F2 + second_term)

    corr_diag = Amp**2 * (fL**(2*(alpha-1)) * fyr**(-2*alpha)) / (2-2*alpha)
    corr_diag *= np.eye(toas.size)

    corr -= np.diag(np.diagonal(corr))
    corr += corr_diag
    return corr

def quantize_fast(toas, toaerrs, flags=None, dt=0.1):
    """
    Function to quantize and average TOAs by observation epoch. Used especially
    for NANOGrav multiband data.

    From https://github.com/vallis/libstempo/blob/master/libstempo/toasim.py

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

def red_noise_powerlaw(A, gamma, freqs):
    """
    Add power law red noise to the prefit residual power spectral density.
    As P=A^2*(f/fyr)^-gamma*df

    Parameters
    ----------
    A : float
        Amplitude of red noise.

    gamma : float
        Spectral index of red noise powerlaw.
    """
    ff = freqs
    # df = np.diff(np.append(np.array([0]),ff))
    return A**2*(ff/fyr)**(-gamma)/(12*np.pi**2) * yr_sec**3 #* df

def S_h(A, alpha, freqs):
    """
    Add power law red noise to the prefit residual power spectral density.
    As S_h=A^2*(f/fyr)^(2*alpha)

    Parameters
    ----------
    A : float
        Amplitude of red noise.

    alpha : float
        Spectral index of red noise powerlaw.
    """
    # ff = freqs
    # df = np.diff(np.append(np.array([0]),ff))
    return A**2*(ff/fyr)**(2*alpha) #* df

def Agwb_from_Seff_plaw(freqs, Tspan, SNR, S_eff, gamma=13/3., alpha=None):
    """
    Must supply numpy.ndarrays.
    """
    if alpha is None:
        alpha = (3-gamma)/2
    else:
        pass

    df = np.diff(np.append(np.array([0]),freqs))

    if hasattr(alpha,'size'):
        fS_sqr = freqs**2 * S_eff**2
        integrand = (freqs[:,np.newaxis]/fyr)**(4*alpha)
        integrand /= fS_sqr[:,np.newaxis]
        fintegral = np.sum(integrand*df[:,np.newaxis],axis=0)
    else:
        integrand = (freqs/fyr)**(4*alpha) / freqs**2 / S_eff**2
        fintegral = np.sum(integrand*df)

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
        quantity = param
    else:
        quantity = param * default_unit

    return quantity
