#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `hasasia` package."""

import pytest
import numpy as np

import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import hasasia.skymap as hsky
import hasasia.forecast as hcast

phi = np.random.uniform(0, 2*np.pi,size=20)
cos_theta = np.random.uniform(-1,1,size=20)
#This ensures a uniform distribution across the sky.
theta = np.arccos(cos_theta)

timespan=[15.0 for ii in range(10)]
timespan.extend([5.0 for ii in range(10)])
freqs = np.logspace(np.log10(5e-10),np.log10(5e-7),500)

A_rn = np.random.uniform(1e-16, 1e-12, size=phi.shape[0])
alphas = np.random.uniform(-3/4, 1, size=phi.shape[0])
sigma = 1e-7

@pytest.fixture
def test_psrs():
    psrs = hsim.sim_pta(timespan=timespan,
        cad=20,
        sigma=sigma,
        phi=phi,
        A_rn=A_rn,
        alpha=alphas,
        theta=theta,
        uneven=False,
        freqs=freqs,
    )
    return psrs

def test_get_sliced_spectra1(test_psrs):
    # Call the function with the mock data
    test_slices = hcast.get_sliced_spectra(
        psrs=test_psrs,
        freqs=freqs,
        start_mjd=0.0,
        end_mjd=11.*365.25,
        min_tspan_cut=3,
        verbose=False,
    )
    assert len(test_slices) == 10
    assert len(test_slices[0].toas) < len(test_psrs[0].toas)
    for p in test_slices:
        tspan = p.toas.max() - p.toas.min() 
        assert tspan > 3*365.25*24*3600
        assert tspan < 11.*365.25*24*3600

def test_get_sliced_spectra2(test_psrs):
    # Call the function with the mock data
    test_slices = hcast.get_sliced_spectra(
        psrs=test_psrs,
        freqs=freqs,
        start_mjd=0.0,
        end_mjd=14.0*365.25,
        min_tspan_cut=3,
        verbose=False,
    )
    assert len(test_slices) == 20
    assert len(test_slices[0].toas) < len(test_psrs[0].toas)
    for p in test_slices:
        tspan = p.toas.max() - p.toas.min() 
        assert tspan > 3*365.25*24*3600
        assert tspan < 14.0*365.25*24*3600
        
def test_change_sigma(test_psrs):
    tp = test_psrs[0]
    original_sigma = tp.toaerrs[0]
    original_ntoa = len(tp.toas)
    tp.change_sigma(
        sigma_factor=0.5,
        start_time=-0.01*365.25, # need to change the sigma at mjd==0
        end_time=15.01*365.25,
        freqs=freqs,
        uneven=False
    )
    assert len(tp.toas) == original_ntoa
    assert tp.toaerrs[0] == original_sigma*0.5


def test_change_cadence(test_psrs):
    tp = test_psrs[0]
    original_sigma = tp.toaerrs[0]
    original_ntoa = len(tp.toas)
    tp.change_cadence(
        cadence=10,
        start_time=-0.01*365.25, # need to change the cadence at mjd==0
        end_time=15.01*365.25,
        freqs=freqs,
        uneven=False,
        
    )
    print(tp.toas[0]/86400/365.25, print(tp.toas[-1]/86400/365.25))
    assert len(tp.toas) == 0.5*original_ntoa
    assert tp.toaerrs[0] == original_sigma
