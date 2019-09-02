#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `hasasia.sensitivity` module and `hasasia.sim` module."""

import pytest
import numpy as np
import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import hasasia.skymap as hsky


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

SM=hsky.SkySensitivity(spectra,theta_gw, phi_gw)
hCirc = hsky.h_circ(1e9,200,5e-9,Tspan,SM.freqs).to('')

SNR = SM.SNR(hCirc.value)

h_divA = (hsky.h_circ(1e9,200,5e-9,Tspan,SM.freqs)
          /hsky.h0_circ(1e9,200,5e-9)).value

Amp = SM.A_gwb(h_divA)
