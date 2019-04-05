# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np
import itertools as it
import scipy.stats as sps

from .sim import create_design_matrix
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
    w = 1/np.sqrt(np.diagonal(N))
    W = np.diag(w)#np.sqrt(1/N)
    # w = np.diagonal(W)

    u, s, v = np.linalg.svd((w * designmatrix.T).T,full_matrices=False)
#     print(u.shape,s.shape,v.shape)
    return np.eye(N.shape[0]) - (1/w * np.dot(u, np.dot(u.T, W)).T).T
    # M = designmatrix
    # n,m = M.shape

    # if N.ndim ==1:
    #     Ninv = np.diag(1/N)
    # else:
    #     Ninv = np.linalg.inv(N)

    # L = np.linalg.cholesky(N)
    # Linv = np.linalg.inv(L)
    # U,s,_ = np.linalg.svd(M, full_matrices=True)
    # Id = np.eye(M.shape[0])
    # S = np.zeros_like(M)
    # S[:m,:m] = np.diag(s)
    # inner = np.linalg.inv(np.matmul(S.T,S))
    # outer = np.matmul(S,np.matmul(inner,S.T))
    #
    # return Id - np.matmul(L,np.matmul(np.matmul(U,outer),np.matmul(U.T,Linv)))

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
           freqs=None, exact_astro_freqs = False, from_G=False,twofreqs=False):
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
        G = G_matrix(M)
        m = G.shape[1]
        if twofreqs:
            Tmat = np.zeros((nf,nf), dtype='float64')
            for ii, f1 in enumerate(ff):
                for jj, f2 in enumerate(ff):
                    Tmat[ii,jj] = np.real(np.sum(np.exp(1j*2*np.pi*(f1*t1-f2*t2))
                                              *np.matmul(G,G.T))/m)
        else:
            for ct, f in enumerate(ff):
                Tmat[ct] = np.real(np.sum(np.exp(-1j*2*np.pi*f*tm)
                                        *np.matmul(G,G.T))/m)

    else:
        R = R_matrix(M, N)
        N_TOA = M.shape[0]
        for ct, f in enumerate(ff):
            Tmat[ct] = np.real(np.sum(np.exp(-1j*2*np.pi*f*tm)*R)/N_TOA)

    return np.real(Tmat), ff, T


def resid_response(freqs):
    """Timing residual response function."""
    return 1/(12*np.pi*freqs**2)

class Pulsar(object):
    """
    Class to encode information about individual pulsars.
    """
    def __init__(self, toas, toaerrs, phi=None, theta=None,
                 designmatrix=None, N=None):
        """ """
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
    def Tf(self):
        if not hasattr(self, '_Tf'):
            self._Tf,_,_ = get_Tf(designmatrix=self.designmatrix,
                                  toas=self.toas, N=self.N,
                                  freqs=self.freqs,**self.Tf_kwargs)
        return self._Tf

    @property
    def Sn(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_Sn'):
            self._Sn = self.psd_prefit/resid_response(self.freqs)/self.Tf
        return self._Sn

    @property
    def S_R(self):
        """Residual power sensitivity. """
        if not hasattr(self, '_Sn'):
            self._Sn = self.psd_prefit/self.Tf
        return self._Sn

    @property
    def h_c(self):
        """Characteristic strain sensitivity"""
        if not hasattr(self, '_h_c'):
            self._h_c = np.sqrt(self.freqs*self.Sn)
        return self._h_c

    @property
    def Omega_gw(self):
        """Energy Density sensitivity"""
        raise NotImplementedError()

    def add_white_noise_power(self, sigma=None, dt=None, vals=False):
        white_noise = 2.0 * dt * (sigma)**2 * np.ones_like(self.freqs)
        self._psd_prefit += white_noise
        if vals:
            return white_noise

    def add_red_noise_power(self, A=None, gamma=None, vals=False):
        """
        Add power law red noise to the prefit residual power spectral density.
        As P=A^2*(f/fyr)^-gamma

        Parameters
        ----------
        A : float
            Amplitude of red noise.

        gamma : float
            Spectral index of red noise noise powerlaw.
        """
        ff = self.freqs
        red_noise = A**2*(ff/fyr)**(-gamma)/(12*np.pi**2) * yr_sec**3
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
        phis = np.array([p.phi for p in spectra])
        thetas = np.array([p.theta for p in spectra])
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

        HDCoff = HellingsDownsCoeff(phis, thetas)
        self.ThetaIJ, self.alphaIJ, self.pairs, self.alphaRSS = HDCoff

        self.T_IJ = np.array([get_TspanIJ(spectra[ii],spectra[jj])
                              for ii,jj in zip(self.pairs[0],self.pairs[1])])
        self.SnI = np.array([sp.Sn for sp in spectra])

    @property
    def S_eff(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_S_eff'):
            ii = self.pairs[0]
            jj = self.pairs[1]
            kk = np.arange(len(self.alphaIJ))
            num = self.T_IJ[kk] / self.Tspan * self.alphaIJ[kk]**2
            series = num[:,np.newaxis]/(self.SnI[ii]*self.SnI[jj])
            self._S_eff = np.power(np.sum(series,axis=0),-0.5)
        return self._S_eff

    @property
    def h_c(self):
        """Characteristic strain sensitivity"""
        if not hasattr(self, '_h_c'):
            self._h_c = np.sqrt(self.freqs*self.S_eff)
        return self._h_c

    @property
    def Omega_gw(self):
        """Energy Density sensitivity"""
        raise NotImplementedError()


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

    # calculate rss (root-sum-squared) of hellings-downs factor
    alphaRSS = np.sqrt(np.sum(alphaIJ**2))
    return np.arccos(cosThetaIJ), alphaIJ, np.array([first,second]), alphaRSS

def get_Tspan(psrs):
    last = np.amax([p.toas.max() for p in psrs])
    first = np.amin([p.toas.min() for p in psrs])
    return last - first

def get_TspanIJ(psr1,psr2):
    start = np.amax([psr1.toas.min(),psr2.toas.min()])
    stop = np.amin([psr1.toas.max(),psr2.toas.max()])
    return stop - start

def SimCurve():
    raise NotImplementedError()
