#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end check of the continuous-wave (CW) normalization.

A circular binary is injected into simulated pulsars using the physical GR
strain (no h0 convention involved), and its optimal matched-filter SNR is
computed directly in the time domain with the timing model projected out.
hasasia's SNR, built from `h0_circ`, must reproduce it.
"""
import numpy as np
import pytest
import healpy as hp
import astropy.units as u
import astropy.constants as c

import hasasia.sim as hsim
import hasasia.sensitivity as hsen
import hasasia.skymap as hsky
import hasasia.utils as hutils

MC, DL = 1e9, 100.0  # chirp mass [Msun], luminosity distance [Mpc]


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(42)
    npsr = 6
    phi_p = rng.uniform(0, 2 * np.pi, npsr)
    theta_p = np.arccos(rng.uniform(-1, 1, npsr))
    psrs = hsim.sim_pta(timespan=10.0, cad=20.0, sigma=1e-7,
                        phi=phi_p, theta=theta_p)
    tspan = hsen.get_Tspan(psrs)
    freqs = np.array([4.0, 30.0]) / tspan
    spectra = [hsen.Spectrum(p, freqs=freqs) for p in psrs]
    theta_gw = np.arccos(rng.uniform(-1, 1, 4))
    phi_gw = rng.uniform(0, 2 * np.pi, 4)
    proj = []
    for p in psrs:
        G = hsen.G_matrix(p.designmatrix)
        proj.append(G @ np.linalg.inv(G.T @ p.N @ G) @ G.T)
    pos = np.array([np.sin(theta_p) * np.cos(phi_p),
                    np.sin(theta_p) * np.sin(phi_p),
                    np.cos(theta_p)]).T
    return dict(psrs=psrs, spectra=spectra, tspan=tspan, freqs=freqs,
                theta_gw=theta_gw, phi_gw=phi_gw, proj=proj, pos=pos)


def _direct_snr2(s, isky, f, cosi, psi, pulsar_term=False):
    """Time-domain matched-filter SNR^2, averaged exactly over initial phase."""
    amp = (4 * (c.G * MC * u.Msun) ** (5 / 3) * (np.pi * f * u.Hz) ** (2 / 3)
           / (c.c ** 4 * DL * u.Mpc)).decompose().value
    a_plus, a_cross = 0.5 * amp * (1 + cosi ** 2), amp * cosi
    k = hsky.khat(s['theta_gw'][isky], s['phi_gw'][isky])
    l = hsky.lhat(s['theta_gw'][isky], s['phi_gw'][isky])
    m = hsky.mhat(s['theta_gw'][isky], s['phi_gw'][isky])
    w = 2 * np.pi * f
    tot = 0.0
    for p, P, n in zip(s['psrs'], s['proj'], s['pos']):
        denom = 1 + n @ k
        fp = 0.5 * ((n @ m) ** 2 - (n @ l) ** 2) / denom
        fx = (n @ l) * (n @ m) / denom
        fp2 = fp * np.cos(2 * psi) + fx * np.sin(2 * psi)
        fx2 = -fp * np.sin(2 * psi) + fx * np.cos(2 * psi)
        delay = (p.pdist / c.c).to('s').value * denom
        for phi0 in (0.0, np.pi / 2):
            def resid(t):
                return (fp2 * a_plus * np.sin(w * t + phi0)
                        - fx2 * a_cross * np.cos(w * t + phi0)) / w
            r = resid(p.toas)
            if pulsar_term:
                r = r - resid(p.toas - delay)
            tot += 0.5 * r @ P @ r
    return tot


def _direct_snr2_avg(s, isky, f, pulsar_term=False):
    """Exact average over cos(inclination) and polarization angle."""
    x, wx = np.polynomial.legendre.leggauss(6)
    psis = np.linspace(0, np.pi, 6, endpoint=False)
    return sum(wi / 2 * _direct_snr2(s, isky, f, xi, ps, pulsar_term) / len(psis)
               for xi, wi in zip(x, wx) for ps in psis)


def test_h0_circ_distance_roundtrip():
    f0 = 1e-8
    h0 = hsky.h0_circ(MC, DL, f0).to('').value
    d = hutils.strain_and_chirp_mass_to_luminosity_distance(h0, MC, f0)
    assert np.isclose(d.value, DL, rtol=1e-10, atol=0)


def test_skymap_snr_matches_injection(setup):
    s = setup
    sky = hsky.SkySensitivity(s['spectra'], s['theta_gw'], s['phi_gw'])
    for fi, f in enumerate(s['freqs']):
        h0 = hsky.h0_circ(MC, DL, f).to('').value
        for isky in range(len(s['theta_gw'])):
            # inclination/polarization averaged
            truth = _direct_snr2_avg(s, isky, f)
            snr = sky.SNR(h0)[fi, isky]
            assert np.isclose(snr ** 2, truth, rtol=1e-3, atol=0)
            # fixed inclination and polarization
            cosi, psi = 0.3, 0.7
            truth = _direct_snr2(s, isky, f, cosi, psi)
            snr = sky.SNR(h0, iota=np.arccos(cosi), psi=psi)[fi, isky]
            assert np.isclose(snr ** 2, truth, rtol=1e-3, atol=0)


def test_explicit_pulsar_term_matches_injection(setup):
    s = setup
    sky = hsky.SkySensitivity(s['spectra'], s['theta_gw'], s['phi_gw'],
                              pulsar_term='explicit')
    for fi, f in enumerate(s['freqs']):
        h0 = hsky.h0_circ(MC, DL, f).to('').value
        for isky in range(len(s['theta_gw'])):
            truth = _direct_snr2_avg(s, isky, f, pulsar_term=True)
            assert np.isclose(sky.SNR(h0)[fi, isky] ** 2, truth, rtol=1e-3, atol=0)


def test_sky_averages_consistent(setup):
    """S_eff_mean and DeterSensitivityCurve must equal the sky average of
    SkySensitivity.S_eff, for both pulsar-term settings."""
    s = setup
    nside = 8
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    for pt in (False, True):
        sky = hsky.SkySensitivity(s['spectra'], theta, phi, pulsar_term=pt)
        det = hsen.DeterSensitivityCurve(s['spectra'], pulsar_term=pt)
        sky_ave = 1 / np.mean(1 / sky.S_eff, axis=1)
        assert np.allclose(sky.S_eff_mean, sky_ave, rtol=1e-12, atol=0)
        assert np.allclose(det.S_eff, sky_ave, rtol=1e-2, atol=0)
    sky_e = hsky.SkySensitivity(s['spectra'], theta, phi, pulsar_term=False)
    sky_p = hsky.SkySensitivity(s['spectra'], theta, phi, pulsar_term=True)
    assert np.allclose(sky_p.S_eff, sky_e.S_eff / 2, rtol=1e-12, atol=0)
