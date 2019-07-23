#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `hasasia` package."""

import pytest
import numpy as np

from hasasia import sensitivity as sens
from hasasia import skymap
from hasasia.sim import sim_pta

phi = np.random.uniform(0, 2*np.pi,size=34)
theta = np.random.uniform(0, np.pi,size=34)
freqs = np.logspace(np.log10(5e-10),np.log10(5e-7),400)
timespans=[11.4 for ii in range(10)]
timespans.extend([3.0 for ii in range(24)])

phi_gw = np.random.uniform(0, 2*np.pi,size=2048)
theta_gw = np.random.uniform(0, np.pi,size=2048)

@pytest.fixture
def pta_simple():
    """
    Sample set of pulsars for testing. All Tspan=11.4 yrs
    """
    return sim_pta(timespan=11.4, cad=23, sigma=1e-7,
                   phi=phi, theta=theta, Npsrs=34)

@pytest.fixture
def pta_heter():
    """
    Sample set of pulsars for testing.
    """
    return sim_pta(timespan=timespans,cad=23,sigma=1e-7,
                   phi=phi,theta=theta)

def test_simple_pta(pta_simple):
    """Sensitivity Tests"""
    spectra = []
    for p in pta_simple:
        sp = sens.Spectrum(p, freqs=freqs)
        sp.Tf
        spectra.append(sp)

    sens.GWBSensitivityCurve(spectra)
    sens.DeterSensitivityCurve(spectra)
    skymap.SkySensitivity(spectra,theta_gw,phi_gw)
