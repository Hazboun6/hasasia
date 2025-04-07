#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `hasasia.sensitivity` module and `hasasia.sim` module."""

import pytest
import numpy as np
import hasasia.sensitivity as hsen
import hasasia.sim as hsim

phi = np.random.uniform(0, 2*np.pi,size=34)
cos_theta = np.random.uniform(-1,1,size=34)
#This ensures a uniform distribution across the sky.
theta = np.arccos(cos_theta)

timespan=[11.4 for ii in range(10)]
timespan.extend([3.0 for ii in range(24)])
freqs = np.logspace(np.log10(5e-10),np.log10(5e-7),500)

A_rn = np.random.uniform(1e-16, 1e-12, size=phi.shape[0])
alphas = np.random.uniform(-3/4, 1, size=phi.shape[0])

@pytest.fixture
def sc_simple():
    '''Test and keep a simple sensitivity curve'''
    psrs = hsim.sim_pta(timespan=timespan, cad=23, sigma=1e-7,
                        phi=phi,theta=theta)

    psr= psrs[0]
    hsen.R_matrix(psr.designmatrix,psr.N)
    hsen.G_matrix(psr.designmatrix)
    hsen.quantize_fast(psr.toas,psr.toaerrs)
    hsen.get_Tf(psr.designmatrix, psr.toas,psr.N,
                from_G=True, exact_astro_freqs=True)

    spectra = []
    for p in psrs:
        sp = hsen.Spectrum(p, freqs=freqs)
        _ = sp.NcalInv
        spectra.append(sp)

    sc1a = hsen.GWBSensitivityCurve(spectra)
    sc1b = hsen.DeterSensitivityCurve(spectra)
    return sc1a, sc1b

def test_sensitivity_w_rednoise():
    '''Test making sensitivity curves with complex RN.'''
    psrs2 = hsim.sim_pta(timespan=timespan,cad=23,sigma=1e-7,
                         phi=phi,theta=theta,
                         A_rn=6e-16,alpha=-2/3.,freqs=freqs)
    psrs3 = hsim.sim_pta(timespan=timespan,cad=23,sigma=1e-7,
                         phi=phi,theta=theta,
                         A_rn=A_rn,alpha=alphas,freqs=freqs)
    spectra2 = []
    for p in psrs2:
        sp = hsen.Spectrum(p, freqs=freqs)
        _ = sp.NcalInv
        spectra2.append(sp)

    spectra3 = []
    for p in psrs3:
        sp = hsen.Spectrum(p, freqs=freqs)
        _ = sp.NcalInv
        spectra3.append(sp)

    spectra2_RRF = []
    for p in psrs2:
        __ = p.K_inv
        sp = hsen.Spectrum(p, freqs=freqs)
        _ = sp.NcalInv
        spectra2_RRF.append(sp)

    spectra3_RRF = []
    for p in psrs3:
        __ = p.K_inv
        sp = hsen.Spectrum(p, freqs=freqs)
        _ = sp.NcalInv
        spectra3_RRF.append(sp)

    spec = spectra3[0]
    spec.Tf
    spec.S_I
    spec.S_R
    spec.h_c
    spec.Omega_gw
    sigma = spec.toaerrs.mean()
    dt = hsen.get_dt(spec.toas).to('s').value
    spec.add_white_noise_power(sigma,dt)
    spec.add_red_noise_power(A=6.8e-16,gamma=13/3.)
    spec.psd_prefit
    spec.psd_postfit
    spec.add_noise_power(spec.psd_prefit)
    hsen.corr_from_psd(freqs, spec.psd_prefit, spec.toas)

    sc2a = hsen.GWBSensitivityCurve(spectra2)
    sc2b = hsen.DeterSensitivityCurve(spectra2)
    sc3a = hsen.GWBSensitivityCurve(spectra3)
    sc3b = hsen.DeterSensitivityCurve(spectra3)

    sc2a.h_c
    sc2a.Omega_gw
    sc2a.S_effIJ
    sc2b.h_c
    sc2b.Omega_gw

    sc3a.h_c
    sc3a.Omega_gw
    sc3a.S_effIJ
    sc3b.h_c
    sc3b.Omega_gw

    spec_RRF = spectra3_RRF[0]
    spec_RRF.Tf
    spec_RRF.S_I
    spec_RRF.S_R
    spec_RRF.h_c
    spec_RRF.Omega_gw
    sigma_RRF = spec.toaerrs.mean()
    dt = hsen.get_dt(spec_RRF.toas).to('s').value
    spec_RRF.add_white_noise_power(sigma_RRF,dt)
    spec_RRF.add_red_noise_power(A=6.8e-16,gamma=13/3.)
    spec_RRF.psd_prefit
    spec_RRF.psd_postfit
    spec_RRF.add_noise_power(spec_RRF.psd_prefit)
    hsen.corr_from_psd(freqs, spec_RRF.psd_prefit, spec_RRF.toas)

    sc2a_RRF = hsen.GWBSensitivityCurve(spectra2_RRF)
    sc2b_RRF = hsen.DeterSensitivityCurve(spectra2_RRF)
    sc3a_RRF = hsen.GWBSensitivityCurve(spectra3_RRF)
    sc3b_RRF = hsen.DeterSensitivityCurve(spectra3_RRF)

    sc2a_RRF.h_c
    sc2a_RRF.Omega_gw
    sc2a_RRF.S_effIJ
    sc2b_RRF.h_c
    sc2b_RRF.Omega_gw

    sc3a_RRF.h_c
    sc3a_RRF.Omega_gw
    sc3a_RRF.S_effIJ
    sc3b_RRF.h_c
    sc3b_RRF.Omega_gw


def test_nonGR():
    psrs = hsim.sim_pta(timespan=timespan, cad=23, sigma=1e-7,
                        phi=phi,theta=theta)
    spectra = []
    for p in psrs:
        sp = hsen.Spectrum(p, freqs=freqs)
        _ = sp.NcalInv
        spectra.append(sp)

    hsen.GWBSensitivityCurve(spectra, orf='st')
    hsen.GWBSensitivityCurve(spectra, orf='dipole')
    hsen.GWBSensitivityCurve(spectra, orf='monopole')

def test_PI_sensitivity(sc_simple):
    # Power Law-Integrated Sensitivity Curves
    sc1a, _ = sc_simple
    hgw = hsen.Agwb_from_Seff_plaw(sc1a.freqs, Tspan=sc1a.Tspan, SNR=3,
                                   S_eff=sc1a.S_eff)
    fyr = 1/(365.25*24*3600)
    plaw_h = hgw*(sc1a.freqs/fyr)**(-2/3)
    PI_sc, plaw = hsen.PI_hc(freqs=sc1a.freqs, Tspan=sc1a.Tspan,
                             SNR=3, S_eff=sc1a.S_eff, N=30)

def test_get_NcalInvIJ():
    psrs = hsim.sim_pta(timespan=timespan, cad=23, sigma=1e-7,
                        phi=phi,theta=theta)
    hsen.get_NcalInvIJ(psrs, 1e-15, freqs)

