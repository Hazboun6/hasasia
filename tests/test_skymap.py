#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `hasasia.sensitivity` module and `hasasia.sim` module."""

import pytest
import numpy as np
import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import hasasia.skymap as hsky

@pytest.fixture
def sm_simple():
    '''Test and keep a simple sensitivity skymap'''
    #Make a set of random sky positions
    phi = np.random.uniform(0, 2*np.pi,size=33)
    cos_theta = np.random.uniform(-1,1,size=33)
    theta = np.arccos(cos_theta)

    #Adding one well-placed sky position for plots.
    phi = np.append(np.array(np.deg2rad(60)),phi)
    theta = np.append(np.array(np.deg2rad(50)),theta)

    #Define the timsespans and TOA errors for the pulsars
    timespans = np.random.uniform(3.0,11.4,size=34)
    Tspan = timespans.max()*365.25*24*3600
    sigma = 1e-7 # 100 ns


    #Simulate a set of identical pulsars, with different sky positions.
    psrs = hsim.sim_pta(timespan=11.4, cad=23, sigma=sigma,
                        phi=phi, theta=theta)



    freqs = np.logspace(np.log10(1/(5*Tspan)),np.log10(2e-7),500)
    spectra = []
    for p in psrs:
        sp = hsen.Spectrum(p, freqs=freqs)
        sp.NcalInv
        spectra.append(sp)


    #Normally use the healpy functions to get the sky coordinates
    #Here just pull random coordinates using numpy to avoid needing healpy
    phi_gw = np.random.uniform(0, 2*np.pi,size=1000)
    cos_theta_gw = np.random.uniform(-1,1,size=1000)
    theta_gw = np.arccos(cos_theta_gw)

    SM = hsky.SkySensitivity(spectra,theta_gw, phi_gw)

    return SM

@pytest.fixture
def spectra_theta_phi():
    '''Test and keep a simple sensitivity skymap'''
    #Make a set of random sky positions
    phi = np.random.uniform(0, 2*np.pi,size=33)
    cos_theta = np.random.uniform(-1,1,size=33)
    theta = np.arccos(cos_theta)

    #Adding one well-placed sky position for plots.
    phi = np.append(np.array(np.deg2rad(60)),phi)
    theta = np.append(np.array(np.deg2rad(50)),theta)

    #Define the timsespans and TOA errors for the pulsars
    timespans = np.random.uniform(3.0,11.4,size=34)
    Tspan = timespans.max()*365.25*24*3600
    sigma = 1e-7 # 100 ns


    #Simulate a set of identical pulsars, with different sky positions.
    psrs = hsim.sim_pta(timespan=11.4, cad=23, sigma=sigma,
                        phi=phi, theta=theta)

    freqs = np.logspace(np.log10(1/(5*Tspan)),np.log10(2e-7),500)
    spectra = []
    for p in psrs:
        sp = hsen.Spectrum(p, freqs=freqs)
        sp.NcalInv
        spectra.append(sp)


    #Normally use the healpy functions to get the sky coordinates
    #Here just pull random coordinates using numpy to avoid needing healpy
    phi_gw = np.random.uniform(0, 2*np.pi,size=1000)
    cos_theta_gw = np.random.uniform(-1,1,size=1000)
    theta_gw = np.arccos(cos_theta_gw)

    return spectra, theta_gw, phi_gw

def test_skymap(sm_simple):
    '''test sky map functionality.'''

    hCirc = hsky.h0_circ(1e9,200,5e-9).to('')

    sm_simple.SNR(hCirc.value)

    sm_simple.h_thresh(2)

    h_divA = (hsky.h_circ(1e9,200,5e-9,sm_simple.Tspan,sm_simple.freqs)
              /hsky.h0_circ(1e9,200,5e-9)).value

    Amp = sm_simple.A_gwb(h_divA)

def test_pulsar_term_skymap(spectra_theta_phi):
    '''scalar test'''
    spectra, theta_gw, phi_gw = spectra_theta_phi
    SM = hsky.SkySensitivity(spectra, theta_gw, phi_gw, pulsar_term=True)

def test_pulsar_term_snr(spectra_theta_phi):
    '''scalar test'''
    spectra, theta_gw, phi_gw = spectra_theta_phi
    SM_pt = hsky.SkySensitivity(spectra, theta_gw, phi_gw, pulsar_term=True)
    SM = hsky.SkySensitivity(spectra, theta_gw, phi_gw, pulsar_term=False)
    hCirc = hsky.h0_circ(1e9,200,5e-9).to('')

    assert (SM_pt.SNR(hCirc.value)/SM.SNR(hCirc.value)/np.sqrt(2)).all()


def test_explicit_pulsar_term_skymap(spectra_theta_phi):
    '''scalar test'''
    spectra, theta_gw, phi_gw = spectra_theta_phi
    SM = hsky.SkySensitivity(spectra, theta_gw, phi_gw, pulsar_term='explicit')

def test_scalar_long_skymap(spectra_theta_phi):
    '''scalar test'''
    spectra, theta_gw, phi_gw = spectra_theta_phi
    SM = hsky.SkySensitivity(spectra, theta_gw, phi_gw, pol='scalar-long')

def test_vector_long_skymap(spectra_theta_phi):
    '''scalar test'''
    spectra, theta_gw, phi_gw = spectra_theta_phi
    SM = hsky.SkySensitivity(spectra, theta_gw, phi_gw, pol='vector-long')

def test_scalar_trans_skymap(spectra_theta_phi):
    '''scalar test'''
    spectra, theta_gw, phi_gw = spectra_theta_phi
    SM = hsky.SkySensitivity(spectra, theta_gw, phi_gw, pol='scalar-trans')
