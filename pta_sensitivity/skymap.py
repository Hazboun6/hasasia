# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np
import astropy.units as u
import astropy.constants as c
# from . import sensitivity as sens
from .sensitivity import SensitivityCurve, resid_response

__all__ = ['',
           '',
           ]

day_sec = 24*3600
yr_sec = 365.25*24*3600


class SkySensitivity(SensitivityCurve):
    '''
    Class to make sky maps for deterministic PTA gravitational wave signals.
    Calculated in terms of $\hat{n}=-\hat{k}$.
    Note: $\hat{l}=> -\hat{l}$ and $\hat{m}=>\hat{m}$.
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
        self.D = num[:,:,:,np.newaxis] / denom[np.newaxis, np.newaxis,:,:]
        self.sky_response = np.einsum('ijkl, ijl ->kl',self.D, self.eplus)**2
        self.sky_response += np.einsum('ijkl, ijl ->kl',self.D, self.ecross)**2
        RNcal = 1.0 / self.SnI
        summand = RNcal.T[:,:,np.newaxis] * self.sky_response[np.newaxis,:,:]
        self.SnSky = 1.0 / np.sum(summand, axis=1)

    def SNR(self, h):
        integrand = 48/5 * h[:,np.newaxis]**2 / self.SnSky
        return np.sqrt(np.trapz(y=integrand, x=self.freqs, axis=0))


def h_circ(M_c, D_L, f):
    return ((4*c.c/(D_L*u.Mpc)) * np.power(c.G*M_c*u.Msun/c.c**3,5/3)
            * np.power(np.pi*f*u.Hz, 2/3))

def khat(theta, phi):
    '''Returns $\hat{k}$ from paper. Also equal to $-\hat{r}=-\hat{n}$.'''
    return np.array([-np.sin(theta)*np.cos(phi),
                     -np.sin(theta)*np.sin(phi),
                     -np.cos(theta)])

def lhat(theta, phi):
    '''Returns $\hat{l}$ from paper. Also equal to $-\hat{phi}$.'''
    return np.array([np.sin(phi), -np.cos(phi), np.zeros_like(theta)])

def mhat(theta, phi):
    '''Returns $\hat{m}$ from paper. Also equal to $-\hat{theta}$.'''
    return np.array([-np.cos(theta)*np.cos(phi),
                     -np.cos(theta)*np.sin(phi),
                     np.sin(theta)])
