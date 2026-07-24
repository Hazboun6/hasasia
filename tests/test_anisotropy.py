#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the `hasasia.anisotropy` module (per-direction anisotropic
sensitivity, the five sky-decomposition bases, and detection statistics)."""

import pytest
import numpy as np
import healpy as hp

import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import hasasia.anisotropy as haniso

# --- small, fast simulated array (module-level, shared across tests) ---
np.random.seed(1234)
NPSR = 12
phi = np.random.uniform(0, 2 * np.pi, size=NPSR)
theta = np.arccos(np.random.uniform(-1, 1, size=NPSR))
freqs = np.logspace(np.log10(5e-10), np.log10(5e-7), 30)
NSIDE = 4
NPIX = hp.nside2npix(NSIDE)
A_GWB = 2.4e-15


@pytest.fixture(scope="module")
def spectra():
    """A small simulated PTA with computed noise spectra."""
    psrs = hsim.sim_pta(timespan=15.0, cad=23, sigma=1e-7,
                        phi=phi, theta=theta, Npsrs=NPSR, freqs=freqs)
    specs = []
    for p in psrs:
        sp = hsen.Spectrum(p, freqs=freqs, amp_gw=A_GWB, gamma_gw=13 / 3.)
        _ = sp.NcalInv
        specs.append(sp)
    return specs


@pytest.fixture(scope="module")
def sky():
    tg, pg = hp.pix2ang(NSIDE, np.arange(NPIX))
    return tg, pg


@pytest.fixture(scope="module")
def ASM(spectra, sky):
    """Pixel-basis Anisotropy object."""
    tg, pg = sky
    return haniso.Anisotropy(spectra, tg, pg, NSIDE=NSIDE, basis='pixel')


# ---------------------------------------------------------------------------
# Construction / basis selection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("basis", ['pixel', 'sph_harm', 'sqrt_sph_harm',
                                   'principal_map'])
def test_construction_all_bases(spectra, sky, basis):
    tg, pg = sky
    A = haniso.Anisotropy(spectra, tg, pg, NSIDE=NSIDE, basis=basis)
    assert A.NPIX == NPIX
    assert A.R_IJ is not None
    assert A.R_IJ.shape[0] == len(A.first)          # one row per pulsar pair
    repr(A)                                          # __repr__ must not raise


def test_unknown_basis_raises(spectra, sky):
    tg, pg = sky
    with pytest.raises(ValueError):
        haniso.Anisotropy(spectra, tg, pg, NSIDE=NSIDE, basis='not_a_basis')


def test_set_basis_rebuilds(ASM):
    R_pix = ASM.R_IJ.copy()
    ASM.set_basis('sph_harm')
    assert ASM.R_IJ.shape[1] != R_pix.shape[1]       # SH has (lmax+1)^2 modes
    ASM.set_basis('pixel')                            # restore for other tests
    assert ASM.R_IJ.shape[1] == NPIX


# ---------------------------------------------------------------------------
# Fisher matrix and effective sensitivity
# ---------------------------------------------------------------------------
def test_Mcal_shapes_and_symmetry(ASM):
    M_diag = ASM.Mcal(diag=True)
    assert M_diag.shape == (len(ASM.freqs), NPIX)
    assert np.all(M_diag >= 0)
    M_full = ASM.Mcal(diag=False, freqs_idx=[len(ASM.freqs) // 2])
    assert M_full.shape == (1, NPIX, NPIX)
    np.testing.assert_allclose(M_full[0], M_full[0].T, rtol=1e-10, atol=0)


def test_S_eff_cauchy_schwarz(ASM):
    """Full-Fisher S_eff >= radiometer S_eff at every pixel (Cauchy-Schwarz)."""
    idx = [len(ASM.freqs) // 2]
    S_rad = ASM.S_eff_fk(diag=True)[idx[0]]
    S_full = ASM.S_eff_fk(diag=False, freqs_idx=idx)[0]
    assert np.all(np.isfinite(S_rad))
    assert np.all(S_full >= S_rad * (1 - 1e-8))


def test_SNR_fk_shapes(ASM):
    Pk = np.ones(NPIX)
    Sh = hsen.S_h(A=A_GWB, alpha=-2 / 3, freqs=ASM.freqs)
    snr = ASM.SNR_fk(Pk, Sh, diag=True)
    assert snr.shape == (len(ASM.freqs), NPIX)
    assert np.all(np.isfinite(snr)) and np.all(snr >= 0)


def test_SNR_total_runs_and_orders(ASM):
    """Full-Fisher total SNR >= radiometer total SNR for an isotropic sky."""
    Pk = np.ones(NPIX)
    Sh = hsen.S_h(A=A_GWB, alpha=-2 / 3, freqs=ASM.freqs)
    snr_rad, _ = ASM.SNR_total(Pk, Sh, diag=True)
    snr_full, _ = ASM.SNR_total(Pk, Sh, diag=False)
    assert np.isfinite(snr_rad) and np.isfinite(snr_full)
    assert snr_full >= snr_rad


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------
def test_injection_vmf(ASM):
    Pk, th, ph, *_ = ASM.injection(NSIDE=NSIDE, theta=np.pi / 2, phi=np.pi,
                                   steepness=10.0, f_iso=0.1, normalize='prob')
    assert Pk.shape == (NPIX,)
    assert np.all(Pk >= 0)
    np.testing.assert_allclose(Pk.sum(), 1.0, rtol=1e-6)        # prob convention


def test_sqrt_sph_harm_positivity(spectra, sky):
    """sqrt-SH reconstruction is non-negative by construction."""
    tg, pg = sky
    A = haniso.Anisotropy(spectra, tg, pg, NSIDE=NSIDE, basis='sqrt_sph_harm')
    rng = np.random.default_rng(0)
    a_LM = rng.standard_normal(A.R_IJ.shape[1])
    Pk = A.basis_strategy.Pk_from_alm(A, a_LM)
    assert Pk.shape == (NPIX,)
    assert np.all(Pk >= 0)


# ---------------------------------------------------------------------------
# Principal-map (Fisher-eigenmap) basis
# ---------------------------------------------------------------------------
def test_principal_modes_match_eigh(spectra, sky):
    """SVD eigenvalues from principal_modes_fk match eigh of the Fisher matrix."""
    tg, pg = sky
    A = haniso.Anisotropy(spectra, tg, pg, NSIDE=NSIDE, basis='principal_map')
    fi = len(A.freqs) // 2
    lam, sig, V = A.principal_modes_fk(freqs_idx=[fi], return_vectors=True)
    M = A.Mcal(diag=False, freqs_idx=[fi])[0]
    eig = np.sort(np.linalg.eigh(M)[0])[::-1]
    K = lam.shape[1]
    np.testing.assert_allclose(lam[0], eig[:K], rtol=1e-6,
                               atol=eig[0] * 1e-10)


def test_principal_map_rad_equals_full(spectra, sky):
    """In the eigenbasis the radiometer and full-Fisher S_eff coincide, and
    S_eff_fk dispatches to S_eff_pm_fk."""
    tg, pg = sky
    A = haniso.Anisotropy(spectra, tg, pg, NSIDE=NSIDE, basis='principal_map')
    fi = len(A.freqs) // 2
    S_pm = A.S_eff_pm_fk(freqs_idx=[fi])[0]
    assert np.all(np.isfinite(S_pm))
    # single value per mode, ascending in mode (decreasing eigenvalue)
    assert np.all(np.diff(S_pm) >= -1e-9 * S_pm[0])
    np.testing.assert_allclose(A.S_eff_fk(freqs_idx=[fi])[0], S_pm)


def test_principal_map_snr_reconstructs_quadratic(spectra, sky):
    """sum_n lambda_n a_n^2 reproduces P^T M P (so the principal-map total SNR
    equals the full-Fisher SNR_total)."""
    tg, pg = sky
    A = haniso.Anisotropy(spectra, tg, pg, NSIDE=NSIDE, basis='principal_map')
    fi = len(A.freqs) // 2
    lam, _, V = A.principal_modes_fk(freqs_idx=[fi], return_vectors=True)
    M = A.Mcal(diag=False, freqs_idx=[fi])[0]
    Pk = np.ones(NPIX)
    a = V[0] @ Pk
    lhs = float(np.sum(lam[0] * a ** 2))
    rhs = float(Pk @ M @ Pk)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-8)


def test_principal_map_skymap(spectra, sky):
    tg, pg = sky
    A = haniso.Anisotropy(spectra, tg, pg, NSIDE=NSIDE, basis='principal_map')
    fi = len(A.freqs) // 2
    m0 = A.principal_map_skymap(0, fi)
    assert m0.shape == (NPIX,)
    assert np.all(np.isfinite(m0))


def test_principal_map_sph_harm_underlying(spectra, sky):
    """The principal-map basis also works on a spherical-harmonic underlying basis."""
    tg, pg = sky
    A = haniso.Anisotropy(spectra, tg, pg, NSIDE=NSIDE,
                          basis='principal_map', underlying='sph_harm')
    fi = len(A.freqs) // 2
    S_pm = A.S_eff_pm_fk(freqs_idx=[fi])[0]
    assert np.any(np.isfinite(S_pm))
    sk = A.principal_map_skymap(0, fi)
    assert sk.shape == (NPIX,)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def test_choose_nside_from_pairs():
    ns = haniso.choose_nside_from_pairs(NPSR * (NPSR - 1) // 2)
    assert ns >= 1 and (ns & (ns - 1)) == 0          # power of two


def test_real_sph_harm_matrix_shape():
    tg, pg = hp.pix2ang(NSIDE, np.arange(NPIX))
    Y, labels = haniso.real_sph_harm_matrix(3, tg, pg)
    assert Y.shape == (16, NPIX)                      # (lmax+1)^2 = 16
    assert len(labels) == 16


# ---------------------------------------------------------------------------
# Map-making, characteristic strain, ORF, and frequency-dependent injection
# ---------------------------------------------------------------------------
def test_S_clean_and_radiometer(ASM):
    Pk = np.ones(NPIX)
    Sh = hsen.S_h(A=A_GWB, alpha=-2 / 3, freqs=ASM.freqs)
    P_clean, sigma, snr = ASM.S_clean(Pk, Sh, diag=True)
    assert P_clean.shape == (len(ASM.freqs), NPIX)
    assert np.all(sigma > 0)
    P_c2, sigma2, snr2 = ASM.radiometer(Pk, Sh)        # diag-True wrapper
    assert sigma2.shape == sigma.shape
    fi = [len(ASM.freqs) // 2]
    _, sigma_full, _ = ASM.S_clean(Pk, Sh, diag=False, freqs_idx=fi)
    assert sigma_full.shape == (1, NPIX)


def test_pixel_orf(ASM):
    Gamma = ASM.basis_strategy.pixel_orf(np.ones(NPIX), normalize='npix')
    assert Gamma.shape == (len(ASM.first),)


def test_injection_multi_hotspot(ASM):
    Pk, *_ = ASM.injection(NSIDE=NSIDE, theta=[np.pi / 2, np.pi / 3],
                           phi=[np.pi / 2, 3 * np.pi / 2],
                           steepness=[10.0, 15.0], amplitudes=[1.0, 0.5],
                           num_of_hotspots=2, f_iso=0.05, normalize='prob')
    assert Pk.shape == (NPIX,)
    assert np.all(Pk >= 0)
    np.testing.assert_allclose(Pk.sum(), 1.0, rtol=1e-6)


def test_injection_frequency_dependent(ASM):
    """Frequency-evolving injection returns a (Nfreqs, NPIX) power array."""
    out = ASM.injection(NSIDE=NSIDE, theta=np.pi / 2, phi=np.pi, steepness=12.0,
                        f_iso=0.1, freqs=ASM.freqs,
                        evolve={'alpha': 0.0, 'beta': 1.0, 'f_iso_gamma': 0.0},
                        normalize='prob')
    P_fk = out[0]
    assert P_fk.shape == (len(ASM.freqs), NPIX)
    assert np.all(P_fk >= 0)


def test_injection_npix_normalization(ASM):
    Pk, *_ = ASM.injection(NSIDE=NSIDE, theta=np.pi / 2, phi=np.pi,
                           steepness=10.0, f_iso=0.0, normalize='npix')
    np.testing.assert_allclose(Pk.sum(), NPIX, rtol=1e-6)


def test_binned_stats():
    x = np.linspace(0.0, 180.0, 300)
    y = np.cos(np.radians(x)) + 0.05 * np.random.standard_normal(x.size)
    edges = np.linspace(0.0, 180.0, 11)
    centers, central, lo, hi = haniso.binned_stats(x, y, bins=edges, stat='mean')
    assert len(centers) == len(central) == 10
    assert np.all(lo <= central + 1e-9) and np.all(hi >= central - 1e-9)
