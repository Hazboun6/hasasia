# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np
import scipy.special as spec
import healpy as hp
import astropy.units as u
import astropy.constants as c
from .sensitivity import DeterSensitivityCurve, resid_response, get_dt
from .utils import strain_and_chirp_mass_to_luminosity_distance

__all__ = ['SkySensitivity',
           'h_circ',
           ]

day_sec = 24*3600
yr_sec = 365.25*24*3600


class SkySensitivity(DeterSensitivityCurve):
    r'''
    Class to make sky maps for deterministic PTA gravitational wave signals.
    Calculated in terms of :math:`\hat{n}=-\hat{k}`.

    Parameters
    ----------
    theta_gw : list, array
        Gravitational wave source sky location colatitude at which to
        calculate sky map.

    phi_gw : list, array
        Gravitational wave source sky location longitude at which to
        calculate sky map.

    pulsar_term : bool, str, optional [True, False, 'explicit']
        Flag for including the pulsar term in sky map sensitivity. True
        includes an idealized factor of two from Equation (36) of `[1]`_.
        The `'explicit'` flag turns on an explicit calculation of
        pulsar terms using pulsar distances. (This option takes
        considerably more computational resources.)

        .. _[1]: https://arxiv.org/abs/1907.04341

    pol: str, optional ['gr','scalar-trans','scalar-long','vector-long']
        Polarization of gravitational waves to be used in pulsar antenna
        patterns. Only one can be used at a time.
    '''
    def __init__(self, spectra, theta_gw, phi_gw,
                 pulsar_term=False, pol='gr', iota=None, psi=None):
        super().__init__(spectra)
        self.pulsar_term = pulsar_term
        self.theta_gw = theta_gw
        self.phi_gw = phi_gw
        self.pos = - khat(self.thetas, self.phis)
        if pulsar_term == 'explicit':
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
        if pulsar_term == 'explicit':
            Dp = self.pdists[:,np.newaxis] * denom
            Dp = self.freqs[:,np.newaxis,np.newaxis] * Dp[np.newaxis,:,:]
            pt = 1-np.exp(-1j*2*np.pi*Dp)
            pt /= 2*np.pi*1j*self.freqs[:,np.newaxis,np.newaxis]
            self.pt_sqr = np.abs(pt)**2

        if pol=='gr':
            self.Fplus = np.einsum('ijkl, ijl ->kl',self.D, self.eplus)
            self.Fcross = np.einsum('ijkl, ijl ->kl',self.D, self.ecross)
            self.sky_response = self.Fplus**2 + self.Fcross**2
        elif pol=='scalar-trans':
            self.Fbreathe = np.einsum('ijkl, ijl ->kl',self.D, self.e_b)
            self.sky_response = self.Fbreathe**2
        elif pol=='scalar-long':
            self.Flong = np.einsum('ijkl, ijl ->kl',self.D, self.e_ell)
            self.sky_response = self.Flong**2
        elif pol=='vector-long':
            self.Fx = np.einsum('ijkl, ijl ->kl',self.D, self.e_x)
            self.Fy = np.einsum('ijkl, ijl ->kl',self.D, self.e_y)
            self.sky_response = self.Fx**2 + self.Fy**2

        if pulsar_term == 'explicit':
            self.sky_response = (0.5 * self.sky_response[np.newaxis,:,:]
                                 * self.pt_sqr)

    def SNR(self, h0, iota=None, psi=None):
        r'''
        Calculate the signal-to-noise ratio of a source given the strain
        amplitude. This is based on Equation (79) from Hazboun, et al., 2019
        `[1]`_.

        .. math::
            \rho(\hat{n})=h_0\sqrt{\frac{T_{\rm obs}}{S_{\rm eff}(f_0 ,\hat{k})}}

        .. _[1]: https://arxiv.org/abs/1907.04341
        '''
        if iota is None and psi is None:
            S_eff = self.S_eff
        elif psi is not None and iota is None:
            raise NotImplementedError('Currently cannot marginalize over phase but not inclination.')
        else:
            S_eff = self.S_eff_full(iota, psi)
        return h0 * np.sqrt(self.Tspan / S_eff)


    def h_thresh(self, SNR=1, iota=None, psi=None):
        r'''
        Method to return a skymap of amplitudes needed to see a circular binary,
        given the specified SNR. This is based on Equation (80) from Hazboun,
        et al., 2019 `[1]`_.

        .. math::
            h_0=\rho(\hat{n})\sqrt{\frac{S_{\rm eff}(f_0 ,\hat{k})}{T_{\rm obs}}}

        .. _[1]: https://arxiv.org/abs/1907.04341


        Parameters
        ----------

        SNR : float, optional
            Desired signal-to-noise ratio.

        Returns
        -------
        An array representing the skymap of amplitudes needed to see the
        given signal with the SNR threshold specified.
        '''
        if iota is None:
            S_eff = self.S_eff
        else:
            S_eff = self.S_eff_full(iota, psi)
        return SNR * np.sqrt(S_eff / self.Tspan)


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

    def sky_response_full(self, iota, psi=None):
        r'''
        Calculate the signal-to-noise ratio of a source given the strain
        amplitude. This is based on Equation (79) from Hazboun, et al., 2019
        `[1]`_, but modified so that you calculate it for a particular inclination angle.

        .. math::
            \rho(\hat{n})=h_0\sqrt{\frac{T_{\rm obs}}{S_{\rm eff}(f_0 ,\hat{k})}}

        .. _[1]: https://arxiv.org/abs/1907.04341
        '''
        if iota is None and psi is not None:
            raise NotImplementedError('Currently cannot marginalize over inclination but not phase.') 
        Ap_sqr = (0.5 * (1 + np.cos(iota)**2))**2
        Ac_sqr = (np.cos(iota))**2
        if psi is None: # case where we average over polarization but not inclination
            # 0.5 is from averaging over polarization
            # Fplus*Fcross term goes to zero
            if isinstance(Ap_sqr,np.ndarray):
                return (self.Fplus[:,:,np.newaxis]**2 + self.Fcross[:,:,np.newaxis]**2)* 0.5 * (Ap_sqr + Ac_sqr)
            elif isinstance(Ap_sqr,(int,float)):
                return (self.Fplus**2 + self.Fcross**2) * 0.5 * (Ac_sqr + Ap_sqr)
        else: # case where we don't average over polarization or inclination
            iota = iota if isinstance(iota, (int,float)) else np.array(iota)
            psi = psi if isinstance(psi, (int,float)) else np.array(psi)
            spsi = np.sin(2*np.array(psi))
            cpsi = np.cos(2*np.array(psi))
            if isinstance(Ap_sqr,np.ndarray) or isinstance(spsi,np.ndarray):
                c1 = Ac_sqr[:,np.newaxis]*spsi**2 + Ap_sqr[:,np.newaxis]*cpsi**2
                c2 = (Ap_sqr[:,np.newaxis] - Ac_sqr[:,np.newaxis]) * cpsi * spsi
                c3 = Ap_sqr[:,np.newaxis]*spsi**2 + Ac_sqr[:,np.newaxis]*cpsi**2
                term1 = self.Fplus[:,:,np.newaxis,np.newaxis]**2 * c1
                term2 = 2 * self.Fplus[:,:,np.newaxis,np.newaxis] * self.Fcross[:,:,np.newaxis,np.newaxis] * c2
                term3 = self.Fcross[:,:,np.newaxis,np.newaxis]**2 * c3
            elif isinstance(Ap_sqr,(int,float)) or isinstance(spsi,(int,float)):
                c1 = Ac_sqr*spsi**2 + Ap_sqr*cpsi**2
                c2 = (Ap_sqr - Ac_sqr) * cpsi * spsi
                c3 = Ap_sqr*spsi**2 + Ac_sqr*cpsi**2
                term1 = self.Fplus**2 * c1
                term2 = 2 * self.Fplus * self.Fcross * c2
                term3 = self.Fcross**2 * c3

            return term1 + term2 + term3

    def S_SkyI_full(self, iota, psi=None):
        """Per Pulsar Strain power sensitivity. """
        t_I = self.T_I / self.Tspan
        RNcalInv = 3.0 * t_I[:,np.newaxis] / self.SnI
        if self.pulsar_term == 'explicit':
            raise NotImplementedError('Currently cannot use pulsar term.')
            # RNcalInv /= resid_response(self.freqs)
            # self._S_SkyI_full = RNcalInv.T[:,:,np.newaxis] * self.sky_response
        else:
            sky_resp = self.sky_response_full(iota, psi)
            if sky_resp.ndim == 2:
                self._S_SkyI_full = (RNcalInv.T[:,:,np.newaxis]
                                     * sky_resp[np.newaxis,:,:])
            elif sky_resp.ndim == 3:
                self._S_SkyI_full = (RNcalInv.T[:,:,np.newaxis,np.newaxis]
                                     * sky_resp[np.newaxis,:,:,:])
            elif sky_resp.ndim == 4:
                self._S_SkyI_full = (RNcalInv.T[:,:,np.newaxis,np.newaxis,np.newaxis]
                                     * sky_resp[np.newaxis,:,:,:,:])

        return self._S_SkyI_full

    def S_eff_full(self, iota, psi=None):
        """
        Strain power sensitivity.
        
        Can calculate margninalized over polarization or not
        with inclination explicit in both cases.
        """
        if self.pulsar_term == 'explicit':
            raise NotImplementedError('Currently cannot use pulsar term.')
            # self._S_eff_full = 1.0 / (4./5 * np.sum(self.S_SkyI, axis=1))
        elif self.pulsar_term:
            raise NotImplementedError('Currently cannot use pulsar term.')
            # self._S_eff_full = 1.0 / (12./5 * np.sum(self.S_SkyI, axis=1))
        else:
            # print(self.S_SkyI_full(iota, psi).shape, self.freqs.size,self.pos.size,self.theta_gw.size)
            self._S_eff_full = 1.0 / np.sum(self.S_SkyI_full(iota, psi), axis=1)

        return self._S_eff_full

    @property
    def S_eff(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_S_eff'):
            if self.pulsar_term == 'explicit':
                self._S_eff = 1.0 / (4./5 * np.sum(self.S_SkyI, axis=1))
            elif self.pulsar_term:
                self._S_eff = 1.0 / (12./5 * np.sum(self.S_SkyI, axis=1))
            else:
                self._S_eff = 1.0 / (6./5 * np.sum(self.S_SkyI, axis=1))
        return self._S_eff

    @property
    def S_SkyI(self):
        """Per Pulsar Strain power sensitivity.
           (Technically, 1 over this is the per pulsar strain power sensitivity.)"""
        if not hasattr(self, '_S_SkyI'):
            t_I = self.T_I / self.Tspan
            RNcalInv = t_I[:,np.newaxis] / self.SnI
            if self.pulsar_term == 'explicit':
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


def calculate_detection_volume(self, f0, SNR_threshold=3.7145, M_c=1e9):
    r"""
    Calculates the detection volume of the PTA
    at a given frequency or list of frequencies.

    Parameters
    ----------
    
    f0 : float
        the frequency [Hz] at which to calculate detection volume
        
    SNR_threshold : float
        the signal to noise to referene detection volume to

    M_c : float
        the chirp mass [Msun] at which to reference detection volume

    Returns
    -------
    
    volume : float
        the detection volume in Mpc^3

    """
    NSIDE = hp.pixelfunc.npix2nside(self.S_eff.shape[1])
    dA = hp.pixelfunc.nside2pixarea(NSIDE, degrees=False)
    if isinstance(f0, (int,float)):
        f_idx = np.array([np.argmin(abs(self.freqs - f0))])
    elif isinstance(f0, (np.ndarray, list)):
        f_idx = np.array([np.argmin(abs(self.freqs - f)) for f in f0])
    h0 = self.h_thresh(SNR=SNR_threshold)
    # detection volume is is the sum of detection radius * pixel area over all pixels
    volume = [dA*np.sum(
        strain_and_chirp_mass_to_luminosity_distance(h0[fdx], M_c, self.freqs[fdx])**3,
        axis=0).value for fdx in f_idx]
    return volume[0] if len(volume)==1 else volume


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
