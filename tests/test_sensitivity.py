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

    sc2a = hsen.GWBSensitivityCurve(spectra2)
    sc2b = hsen.DeterSensitivityCurve(spectra2)
    sc3a = hsen.GWBSensitivityCurve(spectra3)
    sc3b = hsen.DeterSensitivityCurve(spectra3)

def test_PI_sensitivity(sc_simple):
    # Power Law-Integrated Sensitivity Curves
    sc1a, _ = sc_simple
    hgw = hsen.Agwb_from_Seff_plaw(sc1a.freqs, Tspan=sc1a.Tspan, SNR=3,
                                   S_eff=sc1a.S_eff)
    fyr = 1/(365.25*24*3600)
    plaw_h = hgw*(sc1a.freqs/fyr)**(-2/3)
    PI_sc, plaw = hsen.PI_hc(freqs=sc1a.freqs, Tspan=sc1a.Tspan,
                             SNR=3, S_eff=sc1a.S_eff, N=30)
