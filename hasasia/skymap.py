# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np
import scipy.special as spec
import astropy.units as u
import astropy.constants as c
from .sensitivity import DeterSensitivityCurve, resid_response, get_dt

__all__ = ['SkySensitivity',
           'h_circ',
           ]

day_sec = 24*3600
yr_sec = 365.25*24*3600


class SkySensitivity(DeterSensitivityCurve):
    r'''
    Class to make sky maps for deterministic PTA gravitational wave signals.
    Calculated in terms of :math:`\hat{n}=-\hat{k}`.
    '''
    def __init__(self, spectra, theta_gw, phi_gw, pulsar_term=False, pol='gr'):
        super().__init__(spectra)
        self.pulsar_term = pulsar_term
        self.theta_gw = theta_gw
        self.phi_gw = phi_gw
        self.pos = - khat(self.thetas, self.phis)
        if pulsar_term:
            self.pdists = np.array([(sp.pdist/c.c).to('s').value
                                    for sp in spectra]) #pulsar distances

        #Return 3xN array of k,l,m GW position vectors.
        self.K = khat(self.theta_gw, self.phi_gw)
        self.L = lhat(self.theta_gw, self.phi_gw)
        self.M = mhat(self.theta_gw, self.phi_gw)
        LL = np.einsum('ij, kj->ikj', self.L, self.L)
        MM = np.einsum('ij, kj->ikj', self.M, self.M)
        KK = np.einsum('ij, kj->ikj', self.K, self.K)
        LM = np.einsum('ij, kj->ikj', self.L, self.M)
        ML = np.einsum('ij, kj->ikj', self.M, self.L)
        KM = np.einsum('ij, kj->ikj', self.K, self.M)
        MK = np.einsum('ij, kj->ikj', self.M, self.K)
        KL = np.einsum('ij, kj->ikj', self.K, self.L)
        LK = np.einsum('ij, kj->ikj', self.L, self.K)
        self.eplus = MM - LL
        self.ecross = LM + ML
        self.e_b = LL + MM
        self.e_ell = KK # np.sqrt(2)*
        self.e_x = KL + LK
        self.e_y = KM + MK
        num = 0.5 * np.einsum('ij, kj->ikj', self.pos, self.pos)
        denom = 1 + np.einsum('ij, il->jl', self.pos, self.K)

        self.D = num[:,:,:,np.newaxis]/denom[np.newaxis, np.newaxis,:,:]
        if pulsar_term:
            Dp = self.pdists[:,np.newaxis] * denom
            Dp = self.freqs[:,np.newaxis,np.newaxis] * Dp[np.newaxis,:,:]
            pt = 1-np.exp(-1j*2*np.pi*Dp)
            pt /= 2*np.pi*1j*self.freqs[:,np.newaxis,np.newaxis]
            self.pt_sqr = np.abs(pt)**2

        if pol=='gr':
            self.Rplus = np.einsum('ijkl, ijl ->kl',self.D, self.eplus)
            self.Rcross = np.einsum('ijkl, ijl ->kl',self.D, self.ecross)
            self.sky_response = self.Rplus**2 + self.Rcross**2
        elif pol=='scalar-trans':
            self.Rbreathe = np.einsum('ijkl, ijl ->kl',self.D, self.e_b)
            self.sky_response = self.Rbreathe**2
        elif pol=='scalar-long':
            self.Rlong = np.einsum('ijkl, ijl ->kl',self.D, self.e_ell)
            self.sky_response = self.Rlong**2
        elif pol=='vector-long':
            self.Rx = np.einsum('ijkl, ijl ->kl',self.D, self.e_x)
            self.Ry = np.einsum('ijkl, ijl ->kl',self.D, self.e_y)
            self.sky_response = self.Rx**2 + self.Ry**2

        if pulsar_term:
            self.sky_response = (0.5 * self.sky_response[np.newaxis,:,:]
                                 * self.pt_sqr)

    def SNR(self, h):
        integrand = 4.0 * h[:,np.newaxis]**2 / self.S_effSky
        return np.sqrt(np.trapz(y=integrand, x=self.freqs, axis=0))

    def A_gwb(self, h_div_A, SNR=1):
        '''
        Method to return a skymap of amplitudes needed to see signal the
        specified signal, given the specified SNR.

        Parameters
        ----------
        h_div_A : array
            An array that represents the frequency dependence of a signal
            that has been divided by the amplitude. Must cover the same
            frequencies covered by the `Skymap.freqs`.

        SNR : float, optional
            Desired signal-to-noise ratio.

        Returns
        -------
        An array representing the skymap of amplitudes needed to see the
        given signal.
        '''
        integrand = h_div_A[:,np.newaxis]**2 / self.S_eff
        return SNR / np.sqrt(np.trapz(integrand,x=self.freqs,axis=0 ))

    @property
    def S_eff(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_S_eff'):
            if self.pulsar_term:
                self._S_eff = 1.0 / (4./5 * np.sum(self.S_SkyI, axis=1))
            else:
                self._S_eff = 1.0 / (12./5 * np.sum(self.S_SkyI, axis=1))
        return self._S_eff

    @property
    def S_SkyI(self):
        """Per Pulsar Strain power sensitivity. """
        if not hasattr(self, '_S_SkyI'):
            t_I = self.T_I / self.Tspan
            RNcalInv = t_I[:,np.newaxis] / self.SnI
            if self.pulsar_term:
                RNcalInv /= resid_response(self.freqs)
                self._S_SkyI = RNcalInv.T[:,:,np.newaxis] * self.sky_response
            else:
                self._S_SkyI = (RNcalInv.T[:,:,np.newaxis]
                                * self.sky_response[np.newaxis,:,:])

        return self._S_SkyI

    @property
    def S_effSky(self):
        return self.S_eff

    @property
    def h_c(self):
        """Characteristic strain sensitivity"""
        if not hasattr(self, '_h_c'):
            self._h_c = np.sqrt(self.freqs[:,np.newaxis] * self.S_eff)
        return self._h_c

    @property
    def S_eff_mean(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_S_eff_mean'):
            mean_sky = np.mean(np.sum(self.S_SkyI, axis=1), axis=1)
            if self.pulsar_term:
                self._S_eff_mean = 1.0 / (4./5 * mean_sky)
            else:
                self._S_eff_mean = 1.0 / (12./5 * mean_sky)
        return self._S_eff_mean


def h_circ(M_c, D_L, f0, Tspan, f):
    r"""
    Convenience function that returns the Fourier domain representation of a
    single circular super-massive binary black hole.

    Parameters
    ----------

    M_c : float [M_sun]
        Chirp mass of a SMBHB.

    D_L : float [Mpc]
        Luminosity distance to a SMBHB.

    f0 : float [Hz]
        Frequency of the SMBHB.

    Tspan : float [sec]
        Timespan that the binary has been observed. Usually taken as the
        timespan of the data set.

    f : array [Hz]
        Array of frequencies over which to model the Fourier domain signal.

    Returns
    -------

    hcw : array [strain]
        Array of strain values across the frequency range provided for a
        circular SMBHB.

    """
    return h0_circ(M_c, D_L, f0) * Tspan * (np.sinc(Tspan*(f-f0))
                                            + np.sinc(Tspan*(f+f0)))

def h0_circ(M_c, D_L, f0):
    """Amplitude of a circular super-massive binary black hole."""
    return (4*c.c / (D_L * u.Mpc)
            * np.power(c.G * M_c * u.Msun/c.c**3, 5/3)
            * np.power(np.pi * f0 * u.Hz, 2/3))

def khat(theta, phi):
    r'''Returns :math:`\hat{k}` from paper.
    Also equal to :math:`-\hat{r}=-\hat{n}`.'''
    return np.array([-np.sin(theta)*np.cos(phi),
                     -np.sin(theta)*np.sin(phi),
                     -np.cos(theta)])

def lhat(theta, phi):
    r'''Returns :math:`\hat{l}` from paper. Also equal to :math:`-\hat{\phi}`.'''
    return np.array([np.sin(phi), -np.cos(phi), np.zeros_like(theta)])

def mhat(theta, phi):
    r'''Returns :math:`\hat{m}` from paper. Also equal to :math:`-\hat{\theta}`.'''
    return np.array([-np.cos(theta)*np.cos(phi),
                     -np.cos(theta)*np.sin(phi),
                     np.sin(theta)])
