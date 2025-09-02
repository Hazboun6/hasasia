# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np
import scipy.optimize as sopt
import scipy.special as ssp
import scipy.integrate as si
import scipy.stats as ss
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

    def SNR(self, h0, iota=None, psi=None, fidx=None):
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
            raise NotImplementedError('Currently cannot marginalize over polarization angle only.')
        else:
            S_eff = self.S_eff_full(iota, psi)

        if fidx is not None:
            S_eff = S_eff[fidx,:]

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
        if iota is None and psi is None:
            raise ValueError('Must specify sky angle with `*_full`. Use `S_eff` for angle averaged sensitivity.')
        if iota is None and psi is not None:
            raise NotImplementedError('Currently cannot marginalize over inclination but not phase.') 
        Ap_sqr = (1 + np.cos(iota)**2)**2 # (0.5 * (1 + np.cos(iota)**2))**2
        Ac_sqr = (2*np.cos(iota))**2
        if psi is None: # case where we average over polarization but not inclination
            # 0.5 is from averaging over polarization
            # Fplus*Fcross term goes to zero
            if isinstance(Ap_sqr,np.ndarray):
                return (self.Fplus[:,:,np.newaxis]**2 + self.Fcross[:,:,np.newaxis]**2) * 0.5 * (Ap_sqr + Ac_sqr)
            elif isinstance(Ap_sqr,(int,float)):
                return (self.Fplus**2 + self.Fcross**2) * 0.5 * (Ac_sqr + Ap_sqr)
        else: # case where we don't average over polarization or inclination
            c_pluplus, c_pluscross, c_crosscross = self._angle_coefficients(iota=iota, psi=psi)
            if isinstance(Ap_sqr,np.ndarray) or isinstance(psi,np.ndarray):
                term1 = self.Fplus[:,:,np.newaxis,np.newaxis]**2 * c_pluplus
                term2 = 2 * self.Fplus[:,:,np.newaxis,np.newaxis] * self.Fcross[:,:,np.newaxis,np.newaxis] * c_pluscross
                term3 = self.Fcross[:,:,np.newaxis,np.newaxis]**2 * c_crosscross
            elif isinstance(Ap_sqr,(int,float)) or isinstance(psi,(int,float)):
                term1 = self.Fplus**2 * c_pluplus
                term2 = 2 * self.Fplus * self.Fcross * c_pluscross
                term3 = self.Fcross**2 * c_crosscross

            return term1 + term2 + term3
        
    def _angle_coefficients(self, iota, psi=None):
        r'''
        Calculate the signal-to-noise ratio of a source given the strain
        amplitude. This is based on Equation (79) from Hazboun, et al., 2019
        `[1]`_, but modified so that you calculate it for a particular inclination angle.

        .. math::
            \rho(\hat{n})=h_0\sqrt{\frac{T_{\rm obs}}{S_{\rm eff}(f_0 ,\hat{k})}}

        .. _[1]: https://arxiv.org/abs/1907.04341
        '''
        if iota is None and psi is not None:
            raise NotImplementedError('Currently can marginalize over inclination but not phase.') 
        Ap_sqr = (1 + np.cos(iota)**2)**2 # (0.5 * (1 + np.cos(iota)**2))**2
        Ac_sqr = (2*np.cos(iota))**2
        if psi is None: # case where we average over polarization but not inclination
            # 0.5 is from averaging over polarization
            # Fplus*Fcross term goes to zero
            if isinstance(Ap_sqr,np.ndarray):
                return (self.Fplus[:,:,np.newaxis]**2 + self.Fcross[:,:,np.newaxis]**2) * 0.5 * (Ap_sqr + Ac_sqr)
            elif isinstance(Ap_sqr,(int,float)):
                return (self.Fplus**2 + self.Fcross**2) * 0.5 * (Ac_sqr + Ap_sqr)
        else: # case where we don't average over polarization or inclination
            iota = iota if isinstance(iota, (int,float)) else np.array(iota)
            psi = psi if isinstance(psi, (int,float)) else np.array(psi)
            spsi = np.sin(2*np.array(psi))
            cpsi = np.cos(2*np.array(psi))
            if isinstance(Ap_sqr,np.ndarray) or isinstance(spsi,np.ndarray):
                c_pluplus = Ac_sqr[:,np.newaxis]*spsi**2 + Ap_sqr[:,np.newaxis]*cpsi**2
                c_pluscross = (Ap_sqr[:,np.newaxis] - Ac_sqr[:,np.newaxis]) * cpsi * spsi
                c_crosscross = Ap_sqr[:,np.newaxis]*spsi**2 + Ac_sqr[:,np.newaxis]*cpsi**2
            elif isinstance(Ap_sqr,(int,float)) or isinstance(spsi,(int,float)):
                c_pluplus = Ac_sqr*spsi**2 + Ap_sqr*cpsi**2
                c_pluscross = (Ap_sqr - Ac_sqr) * cpsi * spsi
                c_crosscross = Ap_sqr*spsi**2 + Ac_sqr*cpsi**2

            return c_pluplus, c_pluscross, c_crosscross

    def S_SkyI_full(self, iota, psi):
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

    def S_eff_full(self, iota, psi):
        """
        Strain power sensitivity.
        
        Can calculate margninalized over polarization or not
        with inclination explicit in both cases.
        (Use psi=None for marginalization over polarization.)
        
        Params
        ------
        iota : float, array
            Inclination angle of the source.
        psi : float, array
            Polarization angle of the source.

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

    def false_dismissal_prob(self, F_thresh, snr=None, Npsrs=None, ave=None, prob_kwargs={'int_method': 'dblquad','h0':None,'fidx':None}):
        '''
        False dismissal probability of the F-statistic
        Use None for the Fe statistic and the number of pulsars for the Fp stat.
        Note that 1 - false_dismissal_prob is the detection probability.
        
        Parameters
        ----------
        F_thresh : float,array
            F-statistic threshold

        snr : float, optional
            Signal to noise ratio for a given single source.

        Npsrs : string, optional
            Number of pulsars in the array for using the Fp statistic.
            Use None for the Fe statistic.

        ave : string, optional -- choose from ['none','snr','prob']
            Point at which angle averaging happens for the F-statistic.
            See appendix of Baier et al. 2024 `[2]` for details.
                None   -- explicit calculation over inclination and polarization
                'snr'  -- analytic marginalization over snr prior to computing fdp
                'prob' -- numeric marginalization over fdp

        prob_kwargs : dict, optional
            int_method : string, optional -- choose from ['dblquad','trapz']
                Method for integrating over inclination and polarization.
                Use 'dblquad' for scipy integration or 'trapz' for trapezoidal approximation.
            h0 : float, optional
                Strain amplitude of the source.
            fidx : int, optional
                Frequency index of the source.
            snr_grid : array, optional
                Grid of SNR values for the source. Required for 'trapz' method.
                See `sky_ave_SNR_gridded` for gridding convenience.

        Returns
        -------
        fdp : float, array
            False dismissal probability for a given F-statistic threshold.

        .. _[2]: https://arxiv.org/abs/2409.00336
        .._ [3]: https://arxiv.org/abs/1503.04803
        '''
        if Npsrs is None:
            N = 4
        elif isinstance(Npsrs,int):
            N = int(4 * Npsrs)
        
        if ave is None or ave=='none':
            return ss.ncx2.cdf(2*F_thresh, df=N, nc=snr**2)
        elif ave=='snr':
            return ss.chi2.cdf(2*F_thresh, df=N, loc=snr**2)
        elif ave=='prob':
            if prob_kwargs['fidx'] is None:
                raise ValueError('fidx must be set!')
            if prob_kwargs['int_method'] == 'dblquad':
                return self._fdp_angle_averaged_dblquad(F_thresh, prob_kwargs['h0'], prob_kwargs['fidx'])
            elif prob_kwargs['int_method'] == 'trapz':
                return self._fdp_angle_averaged_trapz(prob_kwargs['snr_grid'], F_thresh, prob_kwargs['h0'], prob_kwargs['fidx'])
            else:
                raise ValueError('int_method must be dblquad or trapz')

    def detection_prob(self, F_thresh, snr=None, Npsrs=None, ave=None, prob_kwargs={'h0':None,'fidx':None}):
        '''
        Detection probability of the F-statistic. See Rosado et al. 2015 `[3]` equation 32.
        Use None for the Fe and the number of pulsars for the Fp stat.

       Parameters
        ----------
        F_thresh : float,array
            F-statistic threshold

        snr : float, optional
            Signal to noise ratio for a given single source.

        Npsrs : string, optional
            Number of pulsars in the array for using the Fp statistic.
            Use None for the Fe statistic.

        ave : string, optional -- choose from ['none','snr','prob']
            Point at which angle averaging happens for the F-statistic.
            See appendix of Baier et al. 2024 `[2]` for details.
                None   -- explicit calculation over inclination and polarization
                'snr'  -- analytic marginalization over snr prior to computing fdp
                'prob' -- numeric marginalization over fdp

        prob_kwargs : dict, optional
            int_method : string, optional -- choose from ['dblquad','trapz']
                Method for integrating over inclination and polarization.
                Use 'dblquad' for scipy integration or 'trapz' for trapezoidal approximation.
            h0 : float, optional
                Strain amplitude of the source.
            fidx : int, optional
                Frequency index of the source.
            snr_grid : array, optional
                Grid of SNR values for the source. Required for 'trapz' method.
                See `sky_ave_SNR_gridded` for gridding convenience.

        Returns
        -------
        detection_prob : float, array
            Detection probability for a given F-statistic threshold.

        .. _[2]: https://arxiv.org/abs/2409.00336
        .._ [3]: https://arxiv.org/abs/1503.04803
        '''
        return 1 - self.false_dismissal_prob(F_thresh, snr, Npsrs, ave, prob_kwargs)
    
    def total_detection_probability(self, F_thresh, snr=None, Npsrs=None, ave=None, prob_kwargs={'h0':None, 'fidx': None}):
        '''
        Computes the total detection probability across frequency bins and strain amplitudes of a SMBBH population.
        See equation 33 in Rosado et al. 2015 '[3]' and equation 7 in Baier et al. 2024 '[2]'.
        Can be interpretted as the probability of detecting a single source in *any* frequency bin.
        
       Parameters
        ----------
        F_thresh : float,array
            F-statistic threshold

        snr : float, optional
            Signal to noise ratio for a given single source.

        Npsrs : string, optional
            Number of pulsars in the array for using the Fp statistic.
            Use None for the Fe statistic.

        ave : string, optional -- choose from ['none','snr','prob']
            Point at which angle averaging happens for the F-statistic.
            See appendix of Baier et al. 2024 `[2]` for details.
                None   -- explicit calculation over inclination and polarization
                'snr'  -- analytic marginalization over snr prior to computing fdp
                'prob' -- numeric marginalization over fdp

        prob_kwargs : dict, optional
            int_method : string, optional -- choose from ['dblquad','trapz']
                Method for integrating over inclination and polarization.
                Use 'dblquad' for scipy integration or 'trapz' for trapezoidal approximation.
            h0 : float, optional
                Strain amplitude of the source.
            fidx : int, optional
                Frequency index of the source.
            snr_grid : array, optional
                Grid of SNR values for the source. Required for 'trapz' method.
                See `sky_ave_SNR_gridded` for gridding convenience.

        Returns
        -------
        tdp : float
            Total detection probability for a given F-statistic threshold.

        .. _[2]: https://arxiv.org/abs/2409.00336
        .._ [3]: https://arxiv.org/abs/1503.04803
        '''
        h0s = prob_kwargs['h0']
        fidxs = np.arange(len(prob_kwargs['fidx']))
        return 1. - np.prod(
            [self.false_dismissal_prob(
                F_thresh,
                snr=snr,
                Npsrs=Npsrs,
                ave=ave,
                prob_kwargs={'h0':h0,
                             'fidx':fidx,
                             'int_method':prob_kwargs['int_method'],
                             'snr_grid':prob_kwargs['snr_grid']})
                    for h0, fidx in zip(h0s,fidxs)
            ],
        axis=0)

    def _fdp_angle_averaged_dblquad(self, F_thresh, h0, fidx):
        '''
        The angle-averaged false dismissal probablity. See arXiv....
        '''
        integrand = lambda psi, iota: np.sin(iota)/np.pi*ss.ncx2.cdf(2*F_thresh, df=4, 
                                                                     nc=self.SNR(h0,iota,psi,fidx).mean()**2)
        return si.dblquad(integrand,0,np.pi,-np.pi/4,np.pi/4)[0]

    def _fdp_angle_averaged_trapz(self, snrs, F_thresh, h0, fidx):
        '''
        The angle-averaged false dismissal probablity. See arXiv....
        '''
        # define the points for integration
        iotas = np.linspace(0, np.pi, snrs.shape[0])
        psis = np.linspace(-np.pi/4, np.pi/4, snrs.shape[1])
        # 2d array of integrand values
        Z =  np.sin(iotas)/np.pi*ss.ncx2.cdf(2*F_thresh,
                                            df=4,
                                            nc=(h0*snrs[:, :, fidx])**2)
        # perform the 2D integration using the trapezoidal method twice
        return np.trapz(np.trapz(Z, iotas, axis=1), psis, axis=0)
    
    def sky_ave_SNR_gridded(self, iota, psi, fidxs=None):
        r'''
        Calculate signal-to-noise ratio across a grid of iota and psi values.
        **Note**: This isn't actually SNR but SNR divided by the strain amplitude.
        This allows the values to be used for any signal value.
        '''
        if fidxs is None: # trick so that all the frequencies get used
            fidxs = np.arange(len(self.freqs))
        grid = []
        for _ , iot in enumerate(iota):
            column = []
            for _ , ps in enumerate(psi):
                column.append(
                    np.sqrt(self.Tspan/self.S_eff_full(iot, ps)[fidxs,:]).mean(axis=1)
                    )
            grid.append(column)

        return np.array(grid)

    def _solve_F_given_fdp_snr(self, fdp0=0.05, snr=3, Npsrs=None,):
        """
        Solves for the F-statistic threshold given a false dismissal probability and SNR.
        """
        Npsrs = 1 if Npsrs is None else Npsrs
        F0 = (4*Npsrs+snr**2)/2 
        return sopt.fsolve(lambda F :self.false_dismissal_prob(F, snr, Npsrs=Npsrs, )-fdp0, F0)

    def _solve_snr_given_fdp_F(self, fdp0=0.05, F=3, Npsrs=None,):
        """
        Solves for the signal to noise ration given a false dismissal probability and F-statistic value.
        """
        Npsrs = 1 if Npsrs is None else Npsrs
        snr0 = np.sqrt(2*F-4*Npsrs)
        return sopt.fsolve(lambda snr :self.false_dismissal_prob(F, snr, Npsrs=Npsrs, )-fdp0, snr0)

    @property
    def S_eff(self):
        """
        Strain power sensitivity. NOTE: The prefactors in these expressions are a factor of 4x larger than in 
        Hazboun, et al., 2019 `[1]` due to a redefinition of h0 to match the one in normal use in the PTA community.
        .. _[1]: https://arxiv.org/abs/1907.04341
        """
        if not hasattr(self, '_S_eff'):
            if self.pulsar_term == 'explicit':
                self._S_eff = 1.0 / (16./5 * np.sum(self.S_SkyI, axis=1))
            elif self.pulsar_term:
                self._S_eff = 1.0 / (48./5 * np.sum(self.S_SkyI, axis=1))
            else:
                self._S_eff = 1.0 / (24./5 * np.sum(self.S_SkyI, axis=1))
        return self._S_eff

    @property
    def S_SkyI(self):
        """
        Per Pulsar Strain power sensitivity.
        (Technically, 1 over this is proportional to the per pulsar strain power sensitivity.)
        """
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
        # detection volume is one-third the sum of detection radius * pixel area over all pixels
        volume = [dA/3.*np.sum(
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
