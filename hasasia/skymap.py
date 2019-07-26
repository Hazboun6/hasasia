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
    '''
    Class to make sky maps for deterministic PTA gravitational wave signals.
    Calculated in terms of :math:`\hat{n}=-\hat{k}`.
    '''
    def __init__(self, spectra, theta_gw, phi_gw):
        super().__init__(spectra)
        self.theta_gw = theta_gw
        self.phi_gw = phi_gw
        self.pos = - khat(self.thetas, self.phis)

        #Return 3xN array of k,l,m GW position vectors.
        self.K = khat(self.theta_gw, self.phi_gw)
        self.L = lhat(self.theta_gw, self.phi_gw)
        self.M = mhat(self.theta_gw, self.phi_gw)
        LL = np.einsum('ij, kj->ikj', self.L, self.L)
        MM = np.einsum('ij, kj->ikj', self.M, self.M)
        LM = np.einsum('ij, kj->ikj', self.L, self.M)
        ML = np.einsum('ij, kj->ikj', self.M, self.L)
        self.eplus = LL - MM
        self.ecross = LM + ML
        num = 0.5 * np.einsum('ij, kj->ikj', self.pos, self.pos)
        denom = 1 + np.einsum('ij, il->jl', self.pos, self.K)
        self.D = num[:,:,:,np.newaxis]/denom[np.newaxis, np.newaxis,:,:]
        self.Rplus = np.einsum('ijkl, ijl ->kl',self.D, self.eplus)
        self.Rcross = np.einsum('ijkl, ijl ->kl',self.D, self.ecross)
        self.sky_response = self.Rplus**2 + self.Rcross**2

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
            self._S_eff = 1.0 / (12./5 * np.sum(self.S_SkyI, axis=1))
        return self._S_eff

    @property
    def S_SkyI(self):
        """Per Pulsar Strain power sensitivity. """
        if not hasattr(self, '_S_eff'):
            t_I = self.T_I / self.Tspan
            RNcalInv = t_I[:,np.newaxis] / self.SnI
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


def h_circ(M_c, D_L, f0, Tspan, f):
    """
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
    '''Returns :math:`\hat{k}` from paper.
    Also equal to :math:`-\hat{r}=-\hat{n}`.'''
    return np.array([-np.sin(theta)*np.cos(phi),
                     -np.sin(theta)*np.sin(phi),
                     -np.cos(theta)])

def lhat(theta, phi):
    '''Returns :math:`\hat{l}` from paper. Also equal to :math:`-\hat{\phi}`.'''
    return np.array([np.sin(phi), -np.cos(phi), np.zeros_like(theta)])

def mhat(theta, phi):
    '''Returns :math:`\hat{m}` from paper. Also equal to :math:`-\hat{\theta}`.'''
    return np.array([-np.cos(theta)*np.cos(phi),
                     -np.cos(theta)*np.sin(phi),
                     np.sin(theta)])
