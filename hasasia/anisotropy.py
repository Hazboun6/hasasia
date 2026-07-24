# -*- coding: utf-8 -*-
r"""Anisotropic gravitational-wave-background sensitivity for pulsar timing arrays.

Builds directional effective sensitivities, sensitivity sky maps, and
detection SNR forecasts from a list of `hasasia.Spectrum` objects, via the
per-pulsar-pair Fisher matrix of the cross-correlation statistic.

Naming convention (matching the accompanying paper): the Fisher matrix is
``Mcal``, the paper's calligraphic operator :math:`\mathcal{M}(\hat\Omega,
\hat\Omega';f)` (Eq. 19), spelled out as in ``NcalInv`` for the calligraphic
:math:`\mathcal{N}^{-1}`. In a given basis ``Mcal`` returns the discrete
Fisher matrix indexed by that basis's modes, exactly as the paper writes
them: pixel ``k`` (:math:`M_{kk'}(f)`) in the pixel basis, multipole
``(l, m)`` (:math:`M^{lm,l'm'}(f)`) in the spherical-harmonic bases, and the
eigenmode ``n`` (:math:`M_{\mu\nu}(f)`) in the principal-map basis. The suffix
``_fk`` on the roman-symbol methods (``S_eff_fk``, ``SNR_fk``, ...) gives the
axes of the returned array: frequency ``f`` and the active-basis mode (``k`` /
``(l, m)`` / ``n`` as above). The observing time :math:`T_\mathrm{obs}` is
folded into ``Mcal`` as in the paper (Eq. 19), so ``Mcal`` returns
:math:`\mathcal{M}` directly and the effective sensitivities read
:math:`S_\mathrm{eff}=\sqrt{1/\mathcal{M}}` (Eqs. 20-21) without an explicit
:math:`T_\mathrm{obs}`.
"""
from __future__ import annotations
import numpy as np
from scipy.special import sph_harm
import healpy as hp
import astropy.constants as c
import itertools as it
import warnings
from .sensitivity import GWBSensitivityCurve, get_Tspan



__all__ = ['Anisotropy', 'PixelBasis', 'SphHarmBasis', 'SqrtSHBasis',
           'PrincipalMapBasis',
           'binned_stats', 'choose_nside_from_pairs',
           'real_sph_harm_matrix']

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
  

def _rescale_pk(Pk, normalize, NPIX):
    """
    Return Pk rescaled to the internal 'npix' convention (sum == NPIX).

    If the caller supplies Pk in 'prob' convention (sum == 1), multiply
    by NPIX so the downstream SNR / clean-map math is consistent with
    the Fisher matrix and with the 'npix' literature convention.
    """
    norm = normalize.lower()
    if norm == 'npix':
        return Pk
    if norm == 'prob':
        return Pk * NPIX
    raise ValueError("normalize must be 'npix' or 'prob'")


def binned_stats(x_deg, y, bins=None, stat='mean'):
    """
    Bin ``y`` against ``x_deg`` and return bin centers, central value,
    and lower/upper spread per bin.

    Parameters
    ----------
    x_deg : array_like
        Independent variable in degrees (e.g. pulsar-pair separation).
    y : array_like
        Dependent values with the same shape as ``x_deg``.
    bins : array_like or None
        Bin edges. Defaults to ``np.arange(0, 181, 5)`` (5-degree bins).
    stat : {'mean', 'median'}
        - 'mean'   : central = mean,   spread = ±1σ (std)
        - 'median' : central = median, spread = 16th/84th percentile

    Returns
    -------
    bin_centers, central, lower, upper : ndarrays of shape (len(bins)-1,)
    """
    if bins is None:
        bins = np.arange(0, 181, 5)
    x = np.asarray(x_deg)
    y = np.asarray(y)
    assert x.shape == y.shape

    idx = np.digitize(x, bins, right=False)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    vals, upper, lower = [], [], []
    for b in range(1, len(bins)):
        mask = (idx == b)
        if np.any(mask):
            yb = y[mask]
            if stat == 'mean':
                m = np.mean(yb)
                s = np.std(yb, ddof=1) if yb.size > 1 else 0.0
                vals.append(m); upper.append(m + s); lower.append(m - s)
            elif stat == 'median':
                m = np.median(yb)
                vals.append(m)
                upper.append(np.percentile(yb, 84))
                lower.append(np.percentile(yb, 16))
            else:
                raise ValueError("stat must be 'mean' or 'median'")
        else:
            vals.append(np.nan); upper.append(np.nan); lower.append(np.nan)
    return bin_centers, np.array(vals), np.array(lower), np.array(upper)
  
def choose_nside_from_pairs(num_pairs: int) -> int:
    """
    Pick NSIDE such that NPIX ~ O(num_pairs/24) rounded down to nearest power-of-two NSIDE.
    Your original heuristic mapped to NSIDE = 2**floor(log2(sqrt(num_pairs/24))).
    """
    # guard against small values
    base = max(1, int(np.floor(np.log2(np.sqrt(num_pairs / 24.0)))))
    return 2 ** base


def real_sph_harm_matrix(lmax, theta, phi):
    r"""Real spherical harmonics on a HEALPix grid (Mingarelli et al. 2013).

    Returns the matrix :math:`Y_{lm}(\theta_k, \phi_k)` of real spherical
    harmonics for all :math:`0 \le l \le l_\mathrm{max}` and
    :math:`-l \le m \le l`, in the convention of Mingarelli et al. (2013):
    :math:`Y_{l,-m}` uses :math:`\sqrt{2}\,\mathrm{Im}[Y_l^{|m|}]`,
    :math:`Y_{l,0} = \mathrm{Re}[Y_l^0]`, and :math:`Y_{l,+m}` uses
    :math:`\sqrt{2}\,\mathrm{Re}[Y_l^m]` (Condon-Shortley phase).  These are
    the orthonormal real harmonics (:math:`Y_{00} = 1/\sqrt{4\pi}`), for which
    :math:`c_{00} = \sqrt{4\pi}` yields :math:`P(\hat\Omega) = 1` for an
    isotropic sky.  ``SphHarmBasis`` rescales them to :math:`Y_{00} = 1`
    (:math:`c_{00} = 1` for isotropy); see its docstring.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    theta : ndarray, shape (Npix,)
        Pixel colatitudes in radians.
    phi : ndarray, shape (Npix,)
        Pixel longitudes in radians.

    Returns
    -------
    Y : ndarray, shape (Nlm, Npix)
        Real-valued Y_{lm} evaluated on the input pixel grid, in
        lexicographic (l, m) order with m running from -l to +l.
    labels : list of tuple
        The (l, m) tuple for each row of Y.
    """
    Y_rows = []
    labels = []
    for l in range(lmax + 1):
        for m in range(-l, l+1):
            if m < 0:
                Y_lm = sph_harm(-m, l, phi, theta)   # use |m| and take imag part
                Y_real = np.sqrt(2) * ((-1)**m) * Y_lm.imag
            elif m == 0:
                Y_lm = sph_harm(0, l, phi, theta)
                Y_real = Y_lm.real
            else:  # m > 0
                Y_lm = sph_harm(m, l, phi, theta)
                Y_real = np.sqrt(2) * ((-1)**m) * Y_lm.real

            Y_rows.append(Y_real)
            labels.append((l, m))

    Y = np.vstack(Y_rows)  # (Nlm, Npix)
    return Y, labels


class Basis:
    """Abstract base class for sky-decomposition strategies.

    Concrete subclasses (`PixelBasis`, `SphHarmBasis`, `SqrtSHBasis`) build
    the per-pair response matrix `R_IJ` for a given `Anisotropy` instance.
    Users typically pass the desired basis via the `basis=` argument of
    `Anisotropy(...)` rather than instantiating these classes directly.
    """
    def build_response(self, anisotropy_obj):
        raise NotImplementedError


class PixelBasis(Basis):
    """HEALPix pixel-basis decomposition.

    Decomposes the sky power distribution into NPIX equal-area HEALPix
    pixels (Gorski et al. 2005) and builds the per-pair response
    :math:`R_{IJ}^k = \\mathcal{R}_{IJ}(\\hat\\Omega_k)/N_\\mathrm{pix}` using
    the PTR22 (Pol, Taylor & Romano 2022) convention, where
    :math:`\\mathcal{R}_{IJ}` is the geometric overlap reduction function.
    """

    def __init__(self):
        self.NSIDE = None
        self.NPIX = None
        self.R_IJ = None
        self.P_k = None

    def __repr__(self):
        return f"PixelBasisStrategy(NSIDE={self.NSIDE})" if self.NSIDE is not None else "PixelBasisStrategy(NSIDE=? )"

    def _sync_from_anisotropy(self, A):
        self.NSIDE = A.NSIDE
        self.NPIX = A.NPIX

    def build_response(self, A):
        """
        Build pixel-basis response R_IJ(k) with PTR22 convention:
          R_IJ(Ω_k) = (3/2)(F_a^+F_b^+ + F_a^xF_b^x)
          R_IJ^k     = (1/NPIX) * R_IJ(Ω_k)
        Stores self.R_IJ with shape (Npair, Npix).
        """
        if self.NSIDE is None or self.NPIX is None:
            self._sync_from_anisotropy(A)

        # Antenna patterns per pixel
        Fplus  = np.einsum('ijkl,ijl->kl',  A.D, A.eplus)   # (Npsr, Npix)
        Fcross = np.einsum('ijkl,ijl->kl',  A.D, A.ecross)  # (Npsr, Npix)

        f, s = A.first, A.second

        # R_IJ(Ω_k)
        R = 1.5 * (Fplus[f] * Fplus[s] + Fcross[f] * Fcross[s])  # (Npair, Npix)

        # PTR22 convention pixel response: R_IJ^k
        R_IJ_k = (1.0 / A.NPIX) * R

        # Cache pixel response
        self.R_IJ = R_IJ_k

        if hasattr(A, "P_k"):
            self.P_k = A.P_k

        return {"R_IJ": R_IJ_k}

    def pixel_orf(self, P_k=None, normalize='npix'):
        """
        Effective ORF contraction: Γ_IJ = Σ_k R_IJ^k P_k.

        The cached R_IJ^k carries the (1/NPIX) HEALPix quadrature weight (see
        build_response), so P_k is rescaled to the internal 'npix' convention
        (sum == NPIX) via _rescale_pk before the contraction.  The returned
        Γ_IJ is in physical ORF units: for an isotropic sky it reproduces the
        Hellings-Downs coefficient χ_IJ.

        Parameters
        ----------
        P_k : ndarray (NPIX,) or None
            Sky power distribution.  If None, uses the cached P_k.
        normalize : {'npix', 'prob'}
            Convention of the input P_k.
            'npix' (default): sum(P_k) == NPIX (P == 1 is isotropic).  Used as-is.
            'prob': sum(P_k) == 1.  Rescaled to the 'npix' convention (× NPIX).
        """
        if self.R_IJ is None:
            raise RuntimeError("R_IJ is not built. Call build_response(A) first.")

        if P_k is None:
            if self.P_k is None:
                raise ValueError("No P_k provided and no cached P_k available.")
            P_k = self.P_k

        P_k = np.asarray(P_k, dtype=float)
        if P_k.shape[-1] != self.NPIX:
            raise ValueError(f"P_k has length {P_k.size}, but NPIX={self.NPIX}.")

        # R_IJ carries the (1/NPIX) quadrature weight, so P_k only needs to be
        # in the 'npix' convention before the contraction.
        P_k = _rescale_pk(P_k, normalize, self.NPIX)

        Gamma_IJ = self.R_IJ @ P_k  # (Npair,)
        return Gamma_IJ



class SphHarmBasis(Basis):
    """Real spherical-harmonic basis (Mingarelli et al. 2013).

    Decomposes the sky power as
    :math:`P(\\hat\\Omega) = \\sum_{l,m} c_{lm} Y_{lm}(\\hat\\Omega)`, using
    real spherical harmonics rescaled so that :math:`Y_{00} = 1`
    (:math:`\\sqrt{4\\pi}` times the orthonormal harmonics returned by
    `real_sph_harm_matrix`).  In this normalization :math:`c_{00} = 1`
    recovers :math:`P(\\hat\\Omega) = 1` for an isotropic sky, so the
    monopole mode is the isotropic sky and its effective sensitivity
    coincides with the isotropic limit.
    """
    def __init__(self, lmax=None, kappa=None, use_cached_Y=True):
        """
        Parameters
        ----------
        lmax : int, optional
            Maximum multipole.  If None, defaults to floor(sqrt(N_psrs))
            at build time.
        kappa : float, optional
            GR overall normalization; default 3 / (8 pi).
        use_cached_Y : bool
            Cache Y_lm(theta_k, phi_k) per (NSIDE, lmax) for reuse across
            calls. Default True.
        """
        self.lmax_override = lmax       # may be None
        self._lmax_used = None          # set at build time
        self.kappa = kappa
        self.use_cached_Y = use_cached_Y
        self._Y_cache = {}   # key=(NSIDE,lmax)->(Y, labels)

    def __repr__(self):
        if self.lmax_override is not None:
            return f"SphHarmBasis(lmax={int(self.lmax_override)})"
        if self._lmax_used is not None:
            return f"SphHarmBasis(lmax~={int(self._lmax_used)})"
        return "SphHarmBasis(lmax=? )"

    __str__ = __repr__

    @property
    def lmax(self):
        """Effective lmax (override if provided; otherwise the last used)."""
        return int(self.lmax_override if self.lmax_override is not None else
                   (self._lmax_used if self._lmax_used is not None else -1))

    def _choose_lmax(self, A):
        """Default lmax = floor(sqrt(Npsrs)), with caps and sanity checks."""
        if self.lmax_override is not None:
            lmax = int(self.lmax_override)
        else:
            # Infer pulsar count from A.Npsrs if available, else from A.phis
            Npsrs = getattr(A, "Npsrs", len(getattr(A, "phis", [])))
            lmax = int(np.floor(np.sqrt(Npsrs)))

        # Sanity: non-negative
        lmax = max(lmax, 0)

        # Cap by HEALPix bandlimit for robustness
        lmax_cap = 3 * A.NSIDE - 1
        lmax = min(lmax, lmax_cap)

        return lmax

    def _warn_if_overparameterized(self, A, lmax):
        K = (lmax + 1) ** 2
        Npairs = len(A.first)
        if K > Npairs:
            warnings.warn(
                f"Requested K={(lmax+1)**2} modes exceeds #pairs={Npairs}; "
                "estimates may be poorly conditioned.",
                RuntimeWarning
            )

    def _get_Ylm_matrix(self, A, lmax):
        """
        Return rescaled harmonics Ylm (Nlm, Npix) and labels.
        Uses Ylm = sqrt(4π) * Y_std so that Y_00 = 1,
        following Mingarelli et al. (2013) convention.
        """
        key = (A.NSIDE, lmax)
        if self.use_cached_Y and key in self._Y_cache:
            return self._Y_cache[key]

        pix = np.arange(A.NPIX)
        theta, phi = hp.pix2ang(A.NSIDE, pix)

        # Y_std is orthonormal under ∫ dΩ
        Y_std, labels = real_sph_harm_matrix(lmax, theta, phi)  # (Nlm, Npix)

        # Rescale to Mingarelli convention: Ylm = sqrt(4π) Y_std, so Y_00 = 1
        Ylm = np.sqrt(4.0 * np.pi) * Y_std

        if self.use_cached_Y:
            self._Y_cache[key] = (Ylm, labels)
        return Ylm, labels

    def build_response(self, A):
        pol = getattr(A, 'pol', 'gr')
        if pol != 'gr':
            raise NotImplementedError("SphHarmBasis is currently only implemented for pol='gr'")

        # 1) Antenna patterns per pixel (same as PixelBasis)
        Fplus  = np.einsum('ijkl,ijl->kl', A.D, A.eplus)   # (Npsr, Npix)
        Fcross = np.einsum('ijkl,ijl->kl', A.D, A.ecross)  # (Npsr, Npix)

        # 2) Pair–pixel product ℛ_ab and Convention-B pixel response R_ab^k
        f, s = A.first, A.second
        R_script = 1.5 * (Fplus[f] * Fplus[s] + Fcross[f] * Fcross[s])  # (Npair, Npix)
        R_k = (1.0 / A.NPIX) * R_script                                 # (Npair, Npix)

        # 3) Ylm and projection
        lmax = self._choose_lmax(A)
        self._lmax_used = lmax

        K = (lmax + 1) ** 2
        if K > len(A.first):
            warnings.warn(
                f"Requested K={(lmax+1)**2} modes exceeds #pairs={len(A.first)}; "
                "estimates may be poorly conditioned.",
                RuntimeWarning
            )

        Ylm, labels = self._get_Ylm_matrix(A, lmax)  # (Nlm, Npix)

        # 4) R_ab^{lm} = Σ_k Ylm,lm(Ω_k) * R_ab^k
        R_IJ_lm = R_k @ Ylm.T  # (Npair, Nlm)

        self.R_IJ = R_IJ_lm
        self.labels = labels
        return {"R_IJ": R_IJ_lm, "labels": labels, "lmax_used": lmax}

    def clm_from_Pk(self, A, P_k):
        """
        Project a pixel sky map P_k onto the spherical-harmonic power
        coefficients c_lm (paper Eq. 36) via the inner product (1/NPIX) Σ_k.
        Returns c_lm, labels.
        """
        P_k = np.asarray(P_k, float)
        P_k = P_k / P_k.sum()
        lmax = self._lmax_used if self._lmax_used is not None else self._choose_lmax(A)
        Ylm, labels = self._get_Ylm_matrix(A, lmax)
        c_lm = (1.0 / A.NPIX) * (Ylm @ P_k)
        return c_lm, labels


class SqrtSHBasis(SphHarmBasis):
    """Square-root spherical harmonic basis (Gersbach+2025 Eq. 19,
    Konstandin+2026 Eq. (3.17)).

    Parameterizes sky power as P_k = [Σ_LM a_LM Ylm^LM(Ω_k)]²,
    enforcing P(Ω) ≥ 0 by construction for any {a_LM}.

    The response matrix R_IJ^{lm} is identical to the standard SH basis
    (inherited from SphHarmBasis). The effective angular resolution in P
    is l_max^(P) = 2 L_max^(a), but with only (L_max+1)² free parameters.
    """

    def __repr__(self):
        lmax = self.lmax
        if lmax >= 0:
            return f"SqrtSHBasis(Lmax={lmax})"
        return "SqrtSHBasis(Lmax=?)"

    def Pk_from_alm(self, A, a_LM):
        """Compute pixel-space P_k from sqrt-SH coefficients.

        P_k = [Σ_LM a_LM Ylm^LM(Ω_k)]²

        Parameters
        ----------
        A : Anisotropy
            Parent object (provides NSIDE, NPIX).
        a_LM : ndarray, shape (Nlm,)
            Square-root SH coefficients.

        Returns
        -------
        P_k : ndarray, shape (NPIX,)
            Pixel power map, normalized to sum=1.
        """
        lmax = self._lmax_used if self._lmax_used is not None else self._choose_lmax(A)
        Ylm, _ = self._get_Ylm_matrix(A, lmax)
        q_k = Ylm.T @ a_LM         # (NPIX,) amplitude field
        P_k = q_k ** 2             # non-negative by construction
        total = P_k.sum()
        if total > 0:
            P_k /= total
        return P_k

    def alm_from_Pk(self, A, P_k):
        """Project pixel power map to sqrt-SH coefficients.

        Takes sqrt of P_k, then projects onto Ylm. This is approximate:
        √P_k may not be band-limited to L_max.

        Parameters
        ----------
        A : Anisotropy
            Parent object.
        P_k : ndarray, shape (NPIX,)
            Pixel power map (sum=1, P_k ≥ 0).

        Returns
        -------
        a_LM : ndarray, shape (Nlm,)
        labels : list of (l, m) tuples
        """
        P_k = np.asarray(P_k, float)
        P_k = P_k / P_k.sum()
        lmax = self._lmax_used if self._lmax_used is not None else self._choose_lmax(A)
        Ylm, labels = self._get_Ylm_matrix(A, lmax)
        q_k = np.sqrt(np.clip(P_k, 0.0, None))
        a_LM = (1.0 / A.NPIX) * (Ylm @ q_k)
        return a_LM, labels


class PrincipalMapBasis(Basis):
    """Principal-map (Fisher-eigenmap) basis.

    The principal maps are the eigenmodes of the per-frequency Fisher
    matrix :math:`M(f)`, following Ali-Haimoud, Smith & Mingarelli (2020,
    2021).  In this basis :math:`M` is diagonal, so the radiometer and
    full-Fisher estimators coincide and there is a single effective
    sensitivity per mode,
    :math:`S_\\mathrm{eff}(f,n) = \\sqrt{T_\\mathrm{obs}/\\lambda_n(f)}`,
    where :math:`\\lambda_n` are the eigenvalues of :math:`M(f)`.

    Unlike the fixed-geometry response bases, the eigenmodes are
    frequency-dependent and are derived from an *underlying* basis
    (pixel by default, or spherical harmonic).  Eigendecomposing the
    pixel-basis :math:`M` and keeping the nonzero modes is mathematically
    the observable (pulsar-pair) subspace of Ali-Haimoud et al.; the
    principal maps come out directly as pixel sky maps.  A spherical-
    harmonic underlying basis gives the band-limited variant (the real
    spherical harmonics are the eigenmaps only in the idealized dense,
    isotropic limit; AH2020 Eq. 72).

    ``build_response`` delegates to the underlying basis so the existing
    :meth:`Anisotropy.Mcal` machinery is reused unchanged; the
    eigendecomposition is performed on demand by
    :meth:`Anisotropy.principal_modes_fk`, :meth:`Anisotropy.S_eff_pm_fk`
    and :meth:`Anisotropy.principal_map_skymap`.
    """

    def __init__(self, underlying='pixel', lmax=None):
        self.underlying_name = underlying
        if underlying == 'pixel':
            self.underlying = PixelBasis()
        elif underlying in ('sph_harm', 'sph'):
            self.underlying = SphHarmBasis(lmax=lmax)
        else:
            raise ValueError(
                "PrincipalMapBasis underlying must be 'pixel' or 'sph_harm', "
                f"got {underlying!r}")
        self.labels = None   # integer mode ranks 0..K-1, set at decomposition time

    def __repr__(self):
        return f"PrincipalMapBasis(underlying={self.underlying_name})"

    __str__ = __repr__

    def build_response(self, A):
        """Delegate to the underlying basis so ``Mcal`` uses its response.

        The principal-map eigendecomposition is applied on top of the
        underlying basis's Fisher matrix by the ``Anisotropy`` methods;
        this just builds the underlying ``R_IJ``.
        """
        res = self.underlying.build_response(A)
        # SH/sqrt-SH carry (l,m) labels for reference, but the principal-map
        # mode labels are integer ranks assigned by the eigendecomposition.
        self.labels = res.get("labels", None)
        return res


class Anisotropy(GWBSensitivityCurve):
    """Per-direction anisotropic sensitivity for a pulsar-timing array.

    Computes the Fisher information matrix :math:`M_{kk'}(f)`, the
    per-direction effective sensitivity :math:`S_\\mathrm{eff}(f, \\hat\\Omega)`,
    characteristic strain :math:`h_c(f, \\hat\\Omega)`, and per-pixel and
    total signal-to-noise ratios for an arbitrary anisotropic sky power
    distribution :math:`P(\\hat\\Omega)`.  Inherits from
    `hasasia.sensitivity.GWBSensitivityCurve`.

    Parameters
    ----------
    spectra : list of hasasia.Spectrum
        Per-pulsar Spectrum objects (one per pulsar in the array).
    theta_gw, phi_gw : array-like, shape (NPIX,)
        HEALPix pixel-center coordinates (colatitude, longitude) for the
        GW sky grid.
    basis : {'pixel', 'sph_harm', 'sqrt_sph_harm', 'radiometer'}
        Sky-decomposition basis.  Defaults to 'pixel'.  The basis can be
        changed in place via `set_basis(...)`.
    pol : {'gr'}
        Polarization model.  Currently only general-relativistic
        (transverse-traceless tensor) is supported.
    pulsar_term : {False, 'explicit'}
        If 'explicit', include the pulsar-term contribution using
        per-pulsar distances `pdist`.  Default False (Earth-term only).
    NSIDE : int, optional
        HEALPix `N_side`.  If None, auto-selected from the pair count via
        `choose_nside_from_pairs(...)`.
    lmax : int, optional
        Maximum SH multipole (used by `SphHarmBasis` and `SqrtSHBasis`).
        If None, defaults to floor(sqrt(N_psrs)).

    Notes
    -----
    The principal user-facing methods are `Mcal`, `S_eff_fk`, `SNR_fk`,
    `S_clean`, `radiometer`, `SNR_total`, and `injection`.
    """
    def __init__(self, spectra, theta_gw, phi_gw, basis='pixel', pol='gr', pulsar_term=False, NSIDE=None, lmax=None, underlying='pixel'):
        super().__init__(spectra)
        self.pol = pol
        self.pulsar_term = pulsar_term
        self.theta_gw = theta_gw
        self.phi_gw = phi_gw
        self.pos = - khat(self.thetas, self.phis)
        self.NPIX = hp.nside2npix(2 ** int(np.floor(np.log2(((len(spectra)*(len(spectra)-1))/24)**(1/2)))))
        self.Tspan = get_Tspan(spectra)

        if pulsar_term == 'explicit':
            self.pdists = np.array([(sp.pdist/c.c).to('s').value
                                    for sp in spectra]) # Pulsar distances (light-travel time)

        # Return 3xN array of k, l, m GW position vectors.
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
        num = 0.5 * np.einsum('ij, kj->ikj', self.pos, self.pos)
        denom = 1 + np.einsum('ij, il->jl', self.pos, self.K)
        Npsrs = len(self.phis)
        psr_idx = np.arange(Npsrs)
        pairs = list(it.combinations(psr_idx,2))
    

        # --- HEALPix setup ---
        if NSIDE is None:
            NSIDE = choose_nside_from_pairs(len(pairs))
        self.NSIDE = NSIDE
        self.NPIX = hp.nside2npix(NSIDE)

        # --- Basis selection (pass lmax/underlying through) ---
        self.basis_strategy = self._get_basis(basis, lmax=lmax, underlying=underlying)

        # Outputs
        self.R_IJ = None
        self.lm_labels = None
        
        self.first, self.second = list(map(list, zip(*pairs)))
        self.cosThetaIJ = np.cos(self.thetas[self.first]) * np.cos(self.thetas[self.second]) \
                        + np.sin(self.thetas[self.first]) * np.sin(self.thetas[self.second]) \
                        * np.cos(self.phis[self.first] - self.phis[self.second])

        self.D = num[:,:,:,np.newaxis]/denom[np.newaxis, np.newaxis,:,:]

        # Auto-compute response matrix
        self.compute_response()

    def set_basis(self, basis: str, lmax: int | None = None, underlying: str = 'pixel'):
        """Switch the active sky-decomposition basis and rebuild the response matrix.

        Parameters
        ----------
        basis : {'pixel', 'sph_harm', 'sqrt_sph_harm', 'principal_map'}
            Name of the basis to use for subsequent Fisher / sensitivity
            calculations.
        lmax : int, optional
            Maximum spherical-harmonic multipole (used by the SH and
            sqrt-SH bases, and by the principal-map basis when its
            underlying basis is spherical harmonic).  If None, defaults
            to floor(sqrt(N_psrs)) at build time.
        underlying : {'pixel', 'sph_harm'}, optional
            Underlying basis eigendecomposed by the ``'principal_map'``
            basis (ignored otherwise).  Default ``'pixel'`` (the
            observable-subspace / Ali-Haimoud convention).

        Notes
        -----
        Invalidates any cached basis-dependent attributes (`R_IJ`,
        `Y_lm_k`, `lm_labels`, `_h_c`) and immediately calls
        `compute_response()` to rebuild them in the new basis.
        """
        self.basis_strategy = self._get_basis(basis, lmax=lmax, underlying=underlying)

        # Invalidate basis-dependent results
        self.R_IJ = None
        for attr in ("Y_lm_k", "RIJ_pixel", "Antenna", "lm_labels", "_h_c"):
            if hasattr(self, attr):
                setattr(self, attr, None)

        self.compute_response()

    def _get_basis(self, basis, lmax=None, underlying='pixel'):
        if basis == 'sph_harm':
            return SphHarmBasis(lmax=lmax)
        elif basis == 'sqrt_sph_harm':
            return SqrtSHBasis(lmax=lmax)
        elif basis == 'pixel':
            return PixelBasis()
        elif basis == 'principal_map':
            return PrincipalMapBasis(underlying=underlying, lmax=lmax)
        else:
            raise ValueError(f"Unknown basis: {basis}")

    def compute_response(self):
        """Build the per-pair response matrix `self.R_IJ` in the active basis.

        Delegates to `self.basis_strategy.build_response(self)`.  Called
        automatically by `__init__` and `set_basis`; users typically do
        not need to call this directly.
        """
        for name in ('first', 'second', 'NSIDE', 'NPIX', 'D', 'eplus', 'ecross', 'pol'):
            if not hasattr(self, name):
                raise RuntimeError(f"Missing required attribute '{name}' before response computation.")
        res = self.basis_strategy.build_response(self)
        self.R_IJ = res["R_IJ"]
        self.lm_labels = res.get("labels", None)


    def __repr__(self):
        basis = self.basis_strategy
        basis_name = type(basis).__name__

        # Try extracting key params from the active basis
        details = []
        for attr in ("NSIDE", "nside", "lmax"):
            if hasattr(basis, attr) and getattr(basis, attr) is not None:
                details.append(f"{attr}={getattr(basis, attr)}")
        detail_str = f"{basis_name}({', '.join(details)})" if details else basis_name

        npulsars = getattr(self, "Npsrs", len(getattr(self, "phis", [])) or None)
        if npulsars is not None:
            npairs = npulsars * (npulsars - 1) // 2
            pta_str = f", Npsrs={npulsars}, Npairs={npairs}"
        else:
            pta_str = ""

        return f"Anisotropy(basis={detail_str}, NSIDE={self.NSIDE}{pta_str})"

 
    @property
    def S_IJ(self):
        r"""Per-pair strain noise PSD product, shape (N_pair, N_freq).

        Returns the geometric mean of the per-pulsar noise spectra,
        :math:`S_I(f)\,S_J(f)`, for each pair `(I, J)`, with the same
        frequency grid as `self.freqs`.
        """
        if not hasattr(self, '_S_IJ'):
            ii = self.pairs[0]
            jj = self.pairs[1]
            self._S_IJ =  np.sqrt((self.SnI[ii] * self.SnI[jj]))
        return self._S_IJ


    def _R_phys(self):
        """Physical ORF with basis normalization applied."""
        strat = self.basis_strategy
        if isinstance(strat, PrincipalMapBasis):
            strat = strat.underlying   # normalization follows the underlying basis
        if isinstance(strat, SphHarmBasis):
            return self.R_IJ
        else:
            return self.NPIX * self.R_IJ

    def Mcal(self, diag=True, freqs_idx=None):
        r"""Directional Fisher matrix :math:`\mathcal{M}(\hat\Omega,\hat\Omega';f)` (paper Eq. 19).

        .. math::
            \mathcal{M}_{kk'}(f) = \sum_{I<J} \frac{T_{IJ}}{T_\mathrm{obs}}\,
            \frac{\mathcal{R}_{IJ}(\hat\Omega_k)\,\mathcal{R}_{IJ}(\hat\Omega_{k'})}
            {S_I(f)\,S_J(f)}

        The observing time :math:`T_\mathrm{obs}` is folded in, so this returns
        the paper's calligraphic :math:`\mathcal{M}` directly.  In the active
        basis the returned matrix is indexed by that basis's modes, matching
        the paper: :math:`M_{kk'}(f)` over pixels ``k`` (pixel basis),
        :math:`M^{lm,l'm'}(f)` over multipoles ``(l, m)`` (spherical-harmonic
        bases), and :math:`M_{\mu\nu}(f)` over the underlying modes (eigenmode
        ``n``) in the principal-map basis.

        Uses the :math:`S_I S_J` noise denominator and the physical ORF
        normalization (pixel: ``NPIX*R_IJ``, sph harm: ``R_IJ``).

        Parameters
        ----------
        diag : bool
            If True (default), return the diagonal :math:`\mathcal{M}_{kk}(f)`
            only (fast).  If False, return the full matrix (per-frequency loop).
        freqs_idx : array-like or None
            Frequency indices to compute. None = all frequencies.
            Only used when diag=False.

        Returns
        -------
        ndarray, shape (Nfreqs, Nmodes) if diag=True
        ndarray, shape (Nf_out, Nmodes, Nmodes) if diag=False

        Notes
        -----
        In the ``'principal_map'`` basis this returns the Fisher matrix of
        the *underlying* basis (the matrix that is eigendecomposed), not the
        eigenvalues.  Use :meth:`principal_modes_fk` for the eigenvalues and
        principal maps, and :meth:`S_eff_pm_fk` for the per-mode sensitivity.
        """
        W_if = self.T_IJ[:, None] / (self.Tspan * self.S_IJ**2)
        R_phys = self._R_phys()

        if diag:
            R2 = R_phys**2
            return np.einsum('if,ik->fk', W_if, R2)

        # Full Fisher: M(f) = R^T diag(W_f) R, processed per frequency
        fidxs = np.arange(W_if.shape[1]) if freqs_idx is None else np.asarray(freqs_idx)
        Nmodes = R_phys.shape[1]
        M_full = np.zeros((len(fidxs), Nmodes, Nmodes))

        for fi_out, fi in enumerate(fidxs):
            R_w = R_phys * np.sqrt(W_if[:, fi:fi+1])
            M_full[fi_out] = R_w.T @ R_w

        return M_full

    def Mcal_inv_diag(self, freqs_idx=None, rcond=1e-10):
        r"""Diagonal of the inverse Fisher matrix :math:`[\mathcal{M}^{-1}(f)]_{kk}` per frequency.

        The full-Fisher variance term: :math:`[\mathcal{M}^{-1}(f)]_{kk}`, whose
        square root is the full-Fisher sensitivity (Eq. 21) and, scaled by
        :math:`1/\sqrt{T_\mathrm{obs}}`, the clean-map uncertainty (Eq. 29).
        :math:`T_\mathrm{obs}` is folded into :math:`\mathcal{M}` (see
        :meth:`Mcal`).  Uses an SVD-regularized pseudo-inverse.

        Parameters
        ----------
        freqs_idx : array-like or None
            Frequency indices to compute. None = all frequencies.
        rcond : float
            Regularization: singular values below rcond * max(s) are
            set to zero before inversion.

        Returns
        -------
        ndarray, shape (Nf_out, Nmodes)
            Diagonal of :math:`\mathcal{M}^{-1}` at each frequency.
        """
        # Build and invert one (Nmodes, Nmodes) Fisher matrix at a time
        # rather than materializing the full (Nf, Nmodes, Nmodes) stack,
        # which for many frequencies in the pixel basis can reach several GB.
        W_if = self.T_IJ[:, None] / (self.Tspan * self.S_IJ**2)
        R_phys = self._R_phys()
        fidxs = (np.arange(W_if.shape[1]) if freqs_idx is None
                 else np.atleast_1d(np.asarray(freqs_idx)))
        Nm = R_phys.shape[1]
        result = np.zeros((len(fidxs), Nm))

        for fi_out, fi in enumerate(fidxs):
            R_w = R_phys * np.sqrt(W_if[:, fi:fi+1])
            M_f = R_w.T @ R_w
            U, s, Vt = np.linalg.svd(M_f, hermitian=True)
            s_inv = np.where(s > rcond * s.max(), 1.0 / s, 0.0)
            # diag(M^{-1}) = diag(V S^{-1} V^T) = sum_j V_kj^2 / s_j
            result[fi_out] = ((Vt.T ** 2) @ s_inv)

        return result

    def S_eff_fk(self, diag=True, freqs_idx=None, rcond=1e-10):
        r"""Directional per-mode effective sensitivity.

        The mode axis indexes the active basis, matching the paper: pixel ``k``
        (Eqs. 20/34), multipole ``(l, m)`` (Eqs. 38/39), or eigenmode ``n``.
        With :math:`T_\mathrm{obs}` folded into :math:`\mathcal{M}` (see
        :meth:`Mcal`), the two estimators read directly as in the paper:

        With diag=True (radiometer, Eq. 20):
            :math:`S_\mathrm{eff}^\mathrm{rad}(f) = \sqrt{1/\mathcal{M}_{kk}(f)}`

        With diag=False (full Fisher, Eq. 21):
            :math:`S_\mathrm{eff}^\mathrm{full}(f) = \sqrt{[\mathcal{M}^{-1}]_{kk}(f)}`

        Parameters
        ----------
        diag : bool
            If True, use diagonal Fisher. If False, use full Fisher.
            Ignored in the ``'principal_map'`` basis, where the radiometer
            and full-Fisher estimators coincide.
        freqs_idx : array-like or None
            Frequency indices (only used when diag=False).
        rcond : float
            SVD regularization cutoff (only used when diag=False).

        Notes
        -----
        In the ``'principal_map'`` basis this dispatches to
        :meth:`S_eff_pm_fk`, returning the per-eigenmode sensitivity
        :math:`1/\sqrt{\lambda_n}=\Sigma_n` (Eq. 47); ``diag`` is ignored
        since the two estimators coincide.
        """
        if isinstance(self.basis_strategy, PrincipalMapBasis):
            return self.S_eff_pm_fk(freqs_idx=freqs_idx, rcond=rcond)
        if diag:
            M_kk = self.Mcal(diag=True)
            return np.clip(M_kk, 1e-30, None)**(-1/2)
        else:
            M_inv_kk = self.Mcal_inv_diag(freqs_idx=freqs_idx, rcond=rcond)
            return np.sqrt(np.clip(M_inv_kk, 0.0, None))

    def S_eff_P(self, Pk, normalize='npix', freqs_idx=None):
        r"""Sky-weighted effective sensitivity :math:`S_\mathrm{eff}(f)` for a given sky (paper Eq. 22).

        The effective strain-noise for detecting the specified angular power
        :math:`P(\hat\Omega)`, obtained by contracting the Fisher operator with
        the sky on both sides,

        .. math::
            \frac{1}{S_\mathrm{eff}^2(f)}
            = \iint \frac{d^2\hat\Omega\,d^2\hat\Omega'}{(4\pi)^2}\,
              P(\hat\Omega)\,\mathcal{M}(\hat\Omega,\hat\Omega';f)\,P(\hat\Omega')
            = \sum_{I<J} \frac{T_{IJ}}{T_\mathrm{obs}}\,
              \frac{\Gamma_{IJ}^2(f)}{S_I(f)\,S_J(f)},

        with the sky-weighted overlap reduction function
        :math:`\Gamma_{IJ} = \int (d^2\hat\Omega/4\pi)\,\mathcal{R}_{IJ}(\hat\Omega)P(\hat\Omega)`.
        A single curve for the whole sky ``Pk`` (a function of ``f`` only, no
        mode index), requiring no inversion of the Fisher matrix.

        This is the anisotropic generalization of the isotropic
        :meth:`hasasia.sensitivity.GWBSensitivityCurve.S_eff`: for an isotropic
        sky (``Pk = 1``) it reduces to that curve (paper Eq. 23 / HRS19 Eq. 89),
        and ``SNR^2 = 2 T_obs int (S_h/S_eff)^2 df`` recovers :meth:`SNR_total`
        with ``diag=False``.

        Requires the pixel basis (``Pk`` is a HEALPix sky map).

        Parameters
        ----------
        Pk : ndarray, shape (NPIX,)
            Sky power map.
        normalize : {'npix', 'prob'}
            Convention of ``Pk`` (see :meth:`SNR_fk`).
        freqs_idx : array-like or None
            Frequency indices to compute. None = all frequencies.

        Returns
        -------
        ndarray, shape (Nfreqs,) or (len(freqs_idx),)
            :math:`S_\mathrm{eff}(f)` for the specified sky.
        """
        if not isinstance(self.basis_strategy, PixelBasis):
            raise RuntimeError("S_eff_P requires the pixel basis (Pk is a sky "
                               "map); call set_basis('pixel') first.")
        Pk = _rescale_pk(np.asarray(Pk, dtype=float), normalize, self.NPIX)
        Gamma = self.R_IJ @ Pk                               # Gamma_IJ = int dOmega/4pi R_IJ P
        W = (self.T_IJ / self.Tspan)[:, None] / self.S_IJ**2  # (Npair, Nfreq)
        if freqs_idx is not None:
            W = W[:, np.atleast_1d(freqs_idx)]
        inv_s2 = (Gamma[:, None]**2 * W).sum(axis=0)          # 1/S_eff^2(f)
        return 1.0 / np.sqrt(np.clip(inv_s2, 1e-300, None))

    def principal_modes_fk(self, freqs_idx=None, rcond=1e-10, return_vectors=False):
        r"""Principal maps (Fisher eigenmaps) per frequency.

        Eigendecomposes the per-frequency Fisher matrix
        :math:`M(f) = R_\mathrm{phys}^T W_f R_\mathrm{phys}` of the
        underlying basis via the singular value decomposition of the
        weighted response :math:`R_w = R_\mathrm{phys}\sqrt{W_f}`,
        :math:`W_f = T_{IJ}/(S_I S_J)`.  The squared singular values are
        the Fisher eigenvalues :math:`\lambda_n = s_n^2 = 1/\Sigma_n^2`,
        and the right singular vectors are the principal maps in the
        underlying-basis coordinates.  Modes are ordered most- to
        least-sensitive (decreasing :math:`\lambda`).

        Following Ali-Haimoud, Smith & Mingarelli (2020, 2021).  The
        eigenbasis is frequency-dependent: mode ``n`` is a different sky
        pattern at each frequency.

        Parameters
        ----------
        freqs_idx : array-like or None
            Frequency indices to compute.  None = all frequencies.  The
            per-frequency SVD is the same cost as the full-Fisher path;
            subsample for speed.
        rcond : float
            Singular values below ``rcond * max(s)`` are treated as
            unconstrained (``lambda -> 0``, ``Sigma -> inf``); these span
            the directions orthogonal to the observable subspace.
        return_vectors : bool
            If True, also return the principal-map vectors ``V``.  These
            can be large (``Nf x K x Nmodes``); leave False (default)
            when only the eigenvalue spectrum / sensitivity is needed.

        Returns
        -------
        lambdas : ndarray, shape (Nf, K)
            Fisher eigenvalues per frequency, sorted descending,
            ``K = min(Npair, Nmodes)``.  Below-cutoff modes set to 0.
        Sigma : ndarray, shape (Nf, K)
            Mode noise ``Sigma_n = 1/sqrt(lambda_n)``; ``inf`` below cutoff.
        V : ndarray, shape (Nf, K, Nmodes), optional
            Principal maps in underlying-basis coordinates (rows).  Only
            returned when ``return_vectors=True``.
        """
        W_if = self.T_IJ[:, None] / (self.Tspan * self.S_IJ**2)
        R_phys = self._R_phys()                      # (Npair, Nmodes)
        fidxs = (np.arange(W_if.shape[1]) if freqs_idx is None
                 else np.atleast_1d(np.asarray(freqs_idx)))

        Npair, Nmodes = R_phys.shape
        K = min(Npair, Nmodes)
        lambdas = np.zeros((len(fidxs), K))
        Sigma = np.full((len(fidxs), K), np.inf)
        V = np.zeros((len(fidxs), K, Nmodes)) if return_vectors else None

        for fi_out, fi in enumerate(fidxs):
            R_w = R_phys * np.sqrt(W_if[:, fi:fi+1])             # (Npair, Nmodes)
            _, s, Vt = np.linalg.svd(R_w, full_matrices=False)  # s desc, Vt (K, Nmodes)
            smax = s.max() if s.size else 0.0
            keep = s > rcond * smax if smax > 0 else np.zeros_like(s, dtype=bool)
            lambdas[fi_out] = np.where(keep, s**2, 0.0)
            Sigma[fi_out] = np.where(keep, 1.0 / np.clip(s, 1e-300, None), np.inf)
            if return_vectors:
                V[fi_out] = Vt

        if return_vectors:
            return lambdas, Sigma, V
        return lambdas, Sigma

    def S_eff_pm_fk(self, freqs_idx=None, rcond=1e-10):
        r"""Per-eigenmode effective sensitivity in the principal-map basis.

        :math:`S_\mathrm{eff}(f,n) = 1/\sqrt{\lambda_n(f)} = \Sigma_n(f)`
        (paper Eq. 47), a single value per mode since the radiometer and
        full-Fisher estimators coincide in the Fisher eigenbasis.  The
        eigenvalues :math:`\lambda_n` are those of :math:`\mathcal{M}` with
        :math:`T_\mathrm{obs}` folded in (see :meth:`Mcal`).  Columns are
        ordered most- to least-sensitive; below-cutoff (unconstrained) modes
        return ``inf``.

        Parameters
        ----------
        freqs_idx : array-like or None
            Frequency indices.  None = all frequencies.
        rcond : float
            SVD regularization cutoff (see :meth:`principal_modes_fk`).

        Returns
        -------
        ndarray, shape (Nf, K)
        """
        lambdas, _ = self.principal_modes_fk(freqs_idx=freqs_idx, rcond=rcond)
        with np.errstate(divide='ignore'):
            return np.sqrt(1.0 / lambdas)   # Sigma_n = 1/sqrt(lambda_n); lambda = 0 -> inf

    def principal_map_skymap(self, n, fidx, rcond=1e-10, fix_sign=True):
        r"""Reconstruct principal map ``n`` at frequency index ``fidx`` as a pixel map.

        Returns an ``(NPIX,)`` array suitable for ``healpy.mollview``.  For
        a pixel underlying basis the eigenvector is already a pixel-space
        map; for a spherical-harmonic underlying basis it is recombined
        onto the HEALPix grid via the :math:`Y_{lm}` matrix.

        Parameters
        ----------
        n : int
            Mode rank (0 = most sensitive) at this frequency.
        fidx : int
            Frequency index.
        rcond : float
            SVD regularization cutoff (see :meth:`principal_modes_fk`).
        fix_sign : bool
            Flip the sign so the monopole projection (mean) is >= 0, per
            the Ali-Haimoud convention.  Eigenvectors have an arbitrary
            overall sign.

        Returns
        -------
        ndarray, shape (NPIX,)
        """
        _, _, V = self.principal_modes_fk(freqs_idx=[fidx], rcond=rcond,
                                          return_vectors=True)
        v = V[0, n]                                  # (Nmodes_underlying,)

        strat = self.basis_strategy
        underlying = strat.underlying if isinstance(strat, PrincipalMapBasis) else strat
        if isinstance(underlying, SphHarmBasis):
            lmax = (underlying._lmax_used if underlying._lmax_used is not None
                    else underlying._choose_lmax(self))
            Ylm, _ = underlying._get_Ylm_matrix(self, lmax)   # (Nlm, NPIX)
            pix = Ylm.T @ v
        else:
            pix = v                                  # already a pixel-space vector

        if fix_sign and pix.mean() < 0:
            pix = -pix
        return pix

    def SNR_fk(self, Pk, Sh, diag=True, normalize='npix', freqs_idx=None, rcond=1e-10):
        """
        Per-pixel SNR per frequency bin.

        Returns an array of shape (Nfreqs, Npix) containing the SNR
        (not squared) at each frequency and sky pixel.  Consistent with
        hasasia's convention where SNR methods return the SNR directly.
        The index k labels the modes of the active basis (pixel, (l, m),
        or eigenmode); see the module docstring.

        With diag=True (radiometer, paper Eq. 26 integrand):
            SNR(f, k) = sqrt( (2 T_obs / NPIX^2) * S_h^2(f) * P_k^2 * Mcal_kk(f) )

        With diag=False (full Fisher):
            SNR(f, k) = |S_h(f) P_k| / ( NPIX^2 * sqrt( [Mcal^{-1}]_kk / T_obs ) )

        with T_obs folded into Mcal (see :meth:`Mcal`).  The 1/NPIX^2 arises
        because R_IJ^k = R(Omega_k)/NPIX carries one factor of 1/NPIX, and the
        SNR involves the signal squared.

        Parameters
        ----------
        Pk : ndarray, shape (Npix,)
            Sky power map.
        Sh : ndarray, shape (Nfreqs,)
            Strain power spectral density S_h(f).
        diag : bool
            If True, use diagonal Fisher (radiometer). If False, use
            full Fisher matrix with SVD-regularized inversion.
        normalize : {'npix', 'prob'}
            Convention of the input ``Pk``.
            - 'npix' (default): sum(Pk) == NPIX (literature convention;
              consistent with ``injection(normalize='npix')`` and
              ``Mcal``).
            - 'prob': sum(Pk) == 1 (probability mass); internally
              rescaled by NPIX so the returned SNR matches 'npix'.
        freqs_idx : array-like or None
            Frequency indices (only used when diag=False).
        rcond : float
            SVD regularization cutoff (only used when diag=False).
        """
        Sh = np.asarray(Sh, dtype=float)
        Pk = _rescale_pk(np.asarray(Pk, dtype=float), normalize, self.NPIX)
        if diag:
            M_kk = self.Mcal(diag=True)
            return np.sqrt(2.0 * self.Tspan * (Sh[:, None]**2) * (Pk[None, :]**2) * M_kk / self.NPIX**2)
        else:
            M_inv_kk = self.Mcal_inv_diag(freqs_idx=freqs_idx, rcond=rcond)
            Sh_sel = Sh[freqs_idx] if freqs_idx is not None else Sh
            P_clean = Sh_sel[:, None] * Pk[None, :]
            sigma = np.sqrt(np.clip(M_inv_kk, 0.0, None) / self.Tspan)
            return np.abs(P_clean / np.clip(sigma, 1e-30, None)) / self.NPIX**2

    def S_clean(self, Pk, Sh, diag=True, normalize='npix', freqs_idx=None, rcond=1e-10):
        """
        Forecasted clean map and per-pixel map significance.

        Returns the expected recovered map, per-pixel uncertainty, and
        map significance (P_clean / sigma).  This is the **map-making**
        SNR — how many sigma the estimate at each pixel is above zero —
        NOT the GWB detection SNR.  For detection forecasts, use
        ``SNR_fk`` or ``SNR_total``.

        Follows Grunthal et al. (2026) Eqs. 16-20 and
        NANOGrav 15yr Eq. 19.

        With diag=True (default), this is the **radiometer** estimator
        (diagonal Fisher only).  With diag=False, uses the full Fisher
        matrix with SVD-regularized inversion.

        Formulas (T_obs folded into Mcal; paper Eqs. 29, 43):
            P'_k    = S_h * P_k                            (expected clean map)
            sigma_k = 1/sqrt(T_obs * Mcal_kk)  [diag=True] (per-pixel uncertainty)
                    = sqrt([Mcal^{-1}]_kk / T_obs)  [diag=False]
            snr_k   = P'_k / sigma_k                        (map significance)

        Parameters
        ----------
        Pk : ndarray, shape (Nmodes,)
            Sky power distribution.
        Sh : ndarray, shape (Nfreqs,)
            Strain power spectral density S_h(f).
        diag : bool
            If True, use diagonal Fisher (radiometer). If False, use
            full Fisher matrix with SVD-regularized inversion.
        normalize : {'npix', 'prob'}
            Convention of the input ``Pk``. See ``SNR_fk`` for details.
            'npix' (default) = literature convention (sum=NPIX).
        freqs_idx : array-like or None
            Frequency indices (only used when diag=False).
        rcond : float
            SVD regularization cutoff (only used when diag=False).

        Returns
        -------
        P_clean, sigma, snr : ndarrays, shape (Nfreqs, Nmodes) or (Nf_out, Nmodes)
        """
        Pk = _rescale_pk(np.asarray(Pk, dtype=float), normalize, self.NPIX)
        if diag:
            M_kk = self.Mcal(diag=True)
            P_clean = Sh[:, None] * Pk[None, :]
            sigma = 1.0 / np.sqrt(np.clip(self.Tspan * M_kk, 1e-30, None))
        else:
            M_inv_kk = self.Mcal_inv_diag(freqs_idx=freqs_idx, rcond=rcond)
            Sh_sel = Sh[freqs_idx] if freqs_idx is not None else Sh
            P_clean = Sh_sel[:, None] * Pk[None, :]
            sigma = np.sqrt(np.clip(M_inv_kk, 0.0, None) / self.Tspan)

        snr = P_clean / np.clip(sigma, 1e-30, None)
        return P_clean, sigma, snr

    def radiometer(self, Pk, Sh, normalize='npix'):
        """
        Forecasted radiometer sky map (diagonal Fisher approximation).

        NOTE: This returns the **map-making significance** — how many sigma
        above zero the clean-map estimate is at each pixel.  This is
        different from ``SNR_fk``, which returns the **detection SNR**
        (includes the 1/NPIX^2 normalization for the GWB detection
        statistic).  For detection forecasts, use ``SNR_fk`` or
        ``SNR_total``.

        The radiometer treats each pixel/mode as an independent point
        source, using only the diagonal of the Fisher matrix M_kk.  This
        is the standard PTA sky-map estimator in Konstandin+2026 Eq. (3.16)
        and Chen+2026 Eqs. 17-18, 20.

        In forecasting mode (known signal; T_obs folded into Mcal, paper Eq. 43):
            P_hat_k(f) = S_h(f) * P_k                  (expected recovered map)
            sigma_k(f) = 1 / sqrt(T_obs * Mcal_kk(f))  (per-pixel uncertainty)
            snr_k(f)   = P_hat * sqrt(T_obs * Mcal_kk) (map significance, NOT detection SNR)

        Parameters
        ----------
        Pk : ndarray, shape (Nmodes,)
            Sky power distribution.
        Sh : ndarray, shape (Nfreqs,)
            Strain power spectral density S_h(f).
        normalize : {'npix', 'prob'}
            Convention of the input ``Pk``. See ``SNR_fk`` for details.

        Returns
        -------
        P_hat : ndarray, shape (Nfreqs, Nmodes)
            Expected radiometer sky map estimate.
        sigma : ndarray, shape (Nfreqs, Nmodes)
            Per-pixel/mode uncertainty.
        snr : ndarray, shape (Nfreqs, Nmodes)
            Per-pixel/mode signal-to-noise ratio.
        """
        return self.S_clean(Pk, Sh, diag=True, normalize=normalize)

    def SNR_total(self, Pk, Sh, diag=True, normalize='npix',
                  freqs_idx=None, rcond=1e-10):
        """
        Total detection SNR integrated over frequency and sky.

        Returns (snr_total, snr2_per_freq):
          - snr_total : float — the total SNR (not squared), consistent
            with hasasia's GWBSensitivityCurve.SNR().
          - snr2_per_freq : ndarray — per-frequency SNR^2 contributions
            (squared, for plotting spectral decomposition).

        Two modes:

        1. diag=True (radiometer, paper Eq. 26):
            SNR^2_total = (2 T_obs/NPIX^2) * sum_f S_h^2(f) * sum_k P_k^2 * Mcal_kk(f) * df

        2. diag=False (full Fisher, paper Eq. 25):
            SNR^2_total = (2 T_obs/NPIX^2) * sum_f S_h^2(f) * P^T Mcal(f) P * df
            Uses the full Fisher matrix including off-diagonal
            correlations.  Always >= radiometer (Cauchy-Schwarz).  T_obs is
            folded into Mcal (see :meth:`Mcal`), giving the paper's explicit
            2 T_obs prefactor.

        The 1/NPIX^2 arises because the pixel response R_IJ^k = R(Omega_k)/NPIX
        carries one factor of 1/NPIX, and the SNR involves the signal squared.
        For an isotropic sky (P_k=1) the power is coherent across the whole
        sky, so the matching detection statistic is the FULL-FISHER total
        (diag=False, P^T M P): this exactly recovers the standard hasasia
        isotropic detection SNR (GWBSensitivityCurve.SNR). The diagonal-only
        radiometer (diag=True) is point-source-optimal and undercounts an
        isotropic sky by a large factor, so it does NOT match hasasia.

        Parameters
        ----------
        Pk : ndarray, shape (Npix,)
            Sky power map.
        Sh : ndarray, shape (Nfreqs,)
            Strain power spectral density S_h(f).
        diag : bool
            If True, radiometer total. If False, full Fisher (P^T M P).
        normalize : {'npix', 'prob'}
            Convention of the input ``Pk``.
        freqs_idx : array-like or None
            Frequency indices to use (default: all).
        rcond : float
            SVD regularization cutoff (only used when diag=False).

        Returns
        -------
        snr_total : float
            Total integrated SNR (not squared).  Consistent with
            hasasia's GWBSensitivityCurve.SNR().
        snr2_per_freq : ndarray, shape (Nfreqs,) or (len(freqs_idx),)
            Per-frequency SNR^2 contributions (squared, for spectral
            decomposition plots).
        """
        Sh = np.asarray(Sh, dtype=float)
        Pk = _rescale_pk(np.asarray(Pk, dtype=float), normalize, self.NPIX)

        # Frequency spacing for integration (trapezoid rule)
        f = self.freqs
        if freqs_idx is not None:
            f = f[freqs_idx]
            Sh = Sh[freqs_idx]

        if diag:
            # Radiometer: sum_k P_k^2 * Mcal_kk(f)
            M_kk = self.Mcal(diag=True)
            if freqs_idx is not None:
                M_kk = M_kk[freqs_idx]
            snr2_f = 2.0 * self.Tspan * Sh**2 * np.sum(Pk**2 * M_kk, axis=1) / self.NPIX**2
        else:
            # Full Fisher: P^T Mcal(f) P at each frequency
            nf = len(f)
            snr2_f = np.zeros(nf)
            fidx_list = freqs_idx if freqs_idx is not None else range(len(self.freqs))
            for i, fi in enumerate(fidx_list):
                M_full = self.Mcal(diag=False, freqs_idx=[fi])[0]  # (NPIX, NPIX)
                snr2_f[i] = 2.0 * self.Tspan * Sh[i]**2 * Pk @ M_full @ Pk / self.NPIX**2

        # Integrate over frequency (trapezoid)
        df = np.diff(f)
        snr2_total = np.sum(0.5 * (snr2_f[:-1] + snr2_f[1:]) * df)

        return np.sqrt(snr2_total), snr2_f

    @property
    def h_c(self):
        r"""Directional characteristic-strain sensitivity, shape (N_freq, N_modes).

        Per-mode radiometer characteristic strain (paper Eq. 49),
        :math:`h_c^\mathrm{rad}(f,\hat\Omega_k)=\sqrt{f\,S_\mathrm{eff}^\mathrm{rad}(f,\hat\Omega_k)}`,
        built from the diagonal-Fisher :meth:`S_eff_fk`.  The mode axis follows
        the active basis (pixel ``k`` / multipole ``(l, m)`` / eigenmode ``n``).
        """
        if getattr(self, '_h_c', None) is None:
            self._h_c = np.sqrt(self.freqs[:, np.newaxis] * self.S_eff_fk(diag=True))
        return self._h_c

    def injection(
        self,
        NSIDE,
        theta=None, phi=None,                   # centers (θ=colat, φ=lon) in radians; scalars or arrays
        amplitudes=None,                        # mixture weights at pivot f0; nonnegative (renormalized)
        steepness=None,                         # vMF concentrations κ_i (>0) at pivot f0
        num_of_hotspots=1,
        f_iso=0.0,                              # isotropic fraction at pivot f0 ∈ [0,1]
        rng=None,
        # -------- frequency evolution controls (optional) --------
        freqs=None,                             # None or 1D array of frequencies (Hz)
        evolve=None,                            # dict of power-law/callables/drift rules (see docstring)
        chunk_size_f=None,                      # process frequencies in chunks to reduce peak memory
        # -------- normalization target ----------
        normalize: str = "npix",                # 'npix' (sum=NPIX, literature) or 'prob' (sum=1)
    ):
        """
        Injection of an anisotropic sky using a mixture of von Mises–Fisher (vMF) lobes.

        Base (pivot) model at f=f0:
            P(Ω) = f_iso * (1/NPIX) + (1 - f_iso) * Σ_i w_i * vMF_i(Ω),
        with vMF_i(Ω) ∝ exp(κ_i μ_i · x(Ω)), discretely normalized (sum over pixels equals 1 for the 'hot' part).

        ➤ Normalization (applied last):
            - normalize='npix' (default):  Σ_k P_k = NPIX   (literature form)
            - normalize='prob'           :  Σ_k P_k = 1     (probability mass)

        Frequency evolution (optional):
        If `freqs is None` → returns a single 1D map (P_k).
        If `freqs` is an array → returns P_fk with shape (Nfreqs, Npix); each row normalized per 'normalize'.

        evolve may be:
            - Power-law controls (all optional):
                evolve = dict(
                    f0=1e-8,                       # pivot frequency (Hz); default=median(freqs)
                    alpha=0.0 or (M,),             # weights power-law:  w_i(f) ∝ (f/f0)^alpha_i
                    beta=0.0 or (M,),              # kappa  power-law:  κ_i(f) = κ_i0 * (f/f0)^beta_i
                    f_iso_gamma=0.0,               # f_iso(f) = clip(f_iso0*(f/f0)^gamma, [0,1])
                    kappa_min=1e-3, kappa_max=1e3, # safety bounds for κ
                    drift=None or dict(            # small center drift (radians/Hz)
                        dtheta_df=0.0 or (M,),     # θ(f) = θ0 + (f-f0)*dθ/df
                        dphi_df=0.0   or (M,),     # φ(f) = φ0 + (f-f0)*dφ/df
                    ),
                )
            - Callables (override power-laws if provided):
                evolve = dict(
                    amplitudes_fn = lambda f: (M,)-array of nonnegative weights (renormed per f),
                    kappa_fn      = lambda f: (M,)-array of κ_i(f) > 0,
                    f_iso_fn      = lambda f: scalar in [0,1],
                    drift_fn      = lambda f: (theta_f, phi_f)  # (M,),(M,) or scalars
                )

        Returns
        -------
        If freqs is None:
            P_k : (NPIX,)  with sum = NPIX (normalize='npix') or 1 (normalize='prob')
            theta, phi, w0, kappa0
        If freqs is provided:
            P_fk : (Nfreqs, NPIX)  with per-row sum = NPIX or 1
            theta, phi, w0, kappa0, meta : dict with evolution settings

        Notes
        -----
        • This function is geometry-only; it does not depend on the PTA response or frequency weights.
        • Keep κ moderate relative to NSIDE to avoid under-resolved ultra-narrow lobes.
        """
        if rng is None:
            rng = np.random.default_rng()

        NPIX = hp.nside2npix(NSIDE)
        pix = np.arange(NPIX)
        x_dirs = np.array(hp.pix2vec(NSIDE, pix))  # (3, NPIX)

        # -----------------------------
        # Determine M and broadcast inputs
        # -----------------------------
        M = num_of_hotspots
        for arr in (theta, phi, amplitudes, steepness):
            if arr is not None:
                M = max(M, int(np.size(arr)))
        M = max(M, 1)

        # Centers at pivot
        if theta is None or phi is None:
            cos_th = rng.uniform(-1, 1, size=M)
            theta = np.arccos(cos_th)
            phi   = rng.uniform(0, 2*np.pi, size=M)
        theta = np.atleast_1d(theta).astype(float)
        phi   = np.atleast_1d(phi).astype(float)
        if theta.size == 1 and M > 1: theta = np.full(M, theta.item())
        if phi.size   == 1 and M > 1: phi   = np.full(M, phi.item())
        # Range hygiene (not strict, but helps avoid surprises)
        theta = np.clip(theta, 0.0, np.pi)
        phi   = np.mod(phi, 2*np.pi)

        # κ (steepness) at pivot
        if steepness is None:
            kappa0 = np.full(M, 8.0, dtype=float)
        else:
            kappa0 = np.atleast_1d(steepness).astype(float)
            if kappa0.size == 1 and M > 1:
                kappa0 = np.full(M, kappa0.item())
        if np.any(kappa0 < 0):
            raise ValueError("All κ (steepness) must be non-negative.")

        # Weights at pivot
        if amplitudes is None:
            w0 = np.ones(M, dtype=float)
        else:
            w0 = np.atleast_1d(amplitudes).astype(float)
            if w0.size == 1 and M > 1:
                w0 = np.full(M, w0.item())
        if np.any(w0 < 0):
            raise ValueError("All amplitudes (weights) must be non-negative.")
        w0 = w0 / np.sum(w0)

        # Base centers as unit vectors
        mu0 = np.array(hp.ang2vec(theta, phi))  # (M,3) if M>1 else (3,)
        if mu0.ndim == 1:
            mu0 = mu0[np.newaxis, :]

        # -----------------------------
        # Core mixer for one set (mu, kappa, w)
        # -----------------------------
        def _vmf_mixture(mu, kappa, w):
            # mu: (M,3), kappa: (M,), w: (M,), x_dirs: (3,NPIX)
            d = mu @ x_dirs                     # (M, NPIX)
            s = np.log(w + 1e-300)[:, None] + kappa[:, None] * d
            s_max = np.max(s, axis=0, keepdims=True)
            hot = np.exp(s - s_max).sum(axis=0) * np.exp(s_max.ravel())  # (NPIX,)
            # Normalize 'hot' to sum=1
            if hot.sum() <= 0.0:
                hot = np.ones(NPIX) / NPIX
            else:
                hot /= hot.sum()
            return hot

        # Apply final normalization choice
        def _finish(P_row_prob):
            # P_row_prob sums to 1 here; convert by requested normalization
            if normalize.lower() == "npix":
                return NPIX * P_row_prob
            elif normalize.lower() == "prob":
                return P_row_prob
            else:
                raise ValueError("normalize must be 'npix' or 'prob'")

        # -----------------------------
        # No-frequency path (single map)
        # -----------------------------
        if freqs is None:
            p_hot = _vmf_mixture(mu0, kappa0, w0)
            f_iso0 = float(f_iso)
            if not (0.0 <= f_iso0 <= 1.0):
                raise ValueError("f_iso must be in [0,1].")
            P_prob = f_iso0 * (1.0 / NPIX) + (1.0 - f_iso0) * p_hot
            P_prob /= P_prob.sum()            # exact
            P_k = _finish(P_prob)             # sum=NPIX (default) or 1
            return P_k, theta, phi, w0, kappa0

        # -----------------------------
        # Frequency-dependent path
        # -----------------------------
        freqs = np.asarray(freqs, dtype=float).ravel()
        Nf = freqs.size
        P_fk = np.zeros((Nf, NPIX), dtype=float)

        evo = evolve or {}
        # Pivot frequency
        f0 = float(evo.get("f0", np.median(freqs)))

        # Power-law exponents (broadcast to (M,))
        def _as_M(x, default):
            arr = np.atleast_1d(x if x is not None else default).astype(float)
            if arr.size == 1 and M > 1:
                arr = np.full(M, arr.item())
            return arr

        alpha_M = _as_M(evo.get("alpha", 0.0), 0.0)   # for weights
        beta_M  = _as_M(evo.get("beta",  0.0), 0.0)   # for kappa
        f_iso_gamma = float(evo.get("f_iso_gamma", 0.0))
        kappa_min = float(evo.get("kappa_min", 1e-3))
        kappa_max = float(evo.get("kappa_max", 1e3))

        # Optional drift (linear) or fn override
        drift = evo.get("drift", None)
        dtheta_df = dphi_df = None
        if drift is not None and evo.get("drift_fn", None) is None:
            dtheta_df = _as_M(drift.get("dtheta_df", 0.0), 0.0)
            dphi_df   = _as_M(drift.get("dphi_df",   0.0), 0.0)

        amplitudes_fn = evo.get("amplitudes_fn", None)
        kappa_fn      = evo.get("kappa_fn",      None)
        f_iso_fn      = evo.get("f_iso_fn",      None)
        drift_fn      = evo.get("drift_fn",      None)

        # frequency chunking (memory control)
        if chunk_size_f is None or chunk_size_f <= 0:
            chunk_size_f = Nf

        for start in range(0, Nf, chunk_size_f):
            stop = min(start + chunk_size_f, Nf)
            f_chunk = freqs[start:stop]                    # (Nc,)
            Nc = f_chunk.size
            r = np.clip(f_chunk / f0, 1e-12, None)

            # centers vs frequency
            if drift_fn is not None:
                th_c, ph_c = drift_fn(f_chunk)            # user returns arrays or scalars
                th_c = np.atleast_1d(th_c).astype(float)
                ph_c = np.atleast_1d(ph_c).astype(float)
                if th_c.size == 1 and Nc > 1: th_c = np.full((Nc, M), th_c.item())
                if ph_c.size == 1 and Nc > 1: ph_c = np.full((Nc, M), ph_c.item())
                if th_c.ndim == 1: th_c = np.tile(th_c, (Nc, 1))
                if ph_c.ndim == 1: ph_c = np.tile(ph_c, (Nc, 1))
                mu_f = np.array([hp.ang2vec(th_c[i], ph_c[i]) for i in range(Nc)])  # (Nc,M,3)
            elif (dtheta_df is not None) or (dphi_df is not None):
                th_c = theta[None, :] + (f_chunk[:, None] - f0) * (0.0 if dtheta_df is None else dtheta_df[None, :])
                ph_c = phi[None,   :] + (f_chunk[:, None] - f0) * (0.0 if dphi_df   is None else dphi_df[None, :])
                th_c = np.clip(th_c, 0.0, np.pi)
                ph_c = np.mod(ph_c, 2*np.pi)
                mu_f = np.array([hp.ang2vec(th_c[i], ph_c[i]) for i in range(Nc)])  # (Nc,M,3)
            else:
                mu_f = np.tile(mu0[None, :, :], (Nc, 1, 1))  # (Nc,M,3)

            # κ(f)
            if kappa_fn is not None:
                kappa_f = np.array([np.atleast_1d(kappa_fn(f)).astype(float) for f in f_chunk])
            else:
                kappa_f = (kappa0[None, :] * (r[:, None] ** beta_M[None, :]))
            kappa_f = np.clip(kappa_f, kappa_min, kappa_max)

            # w(f)
            if amplitudes_fn is not None:
                w_f = np.array([np.atleast_1d(amplitudes_fn(f)).astype(float) for f in f_chunk])
            else:
                w_f = (w0[None, :] * (r[:, None] ** alpha_M[None, :]))
            w_f = np.clip(w_f, 0.0, None)
            w_f = w_f / np.maximum(w_f.sum(axis=1, keepdims=True), 1e-300)

            # f_iso(f)
            if f_iso_fn is not None:
                f_iso_f = np.array([float(f_iso_fn(f)) for f in f_chunk])
            else:
                f_iso_f = np.clip(float(f_iso) * (r ** f_iso_gamma), 0.0, 1.0)

            # assemble per frequency
            for j in range(Nc):
                hot = _vmf_mixture(mu_f[j], kappa_f[j], w_f[j])     # sum=1
                P_prob = f_iso_f[j] * (1.0 / NPIX) + (1.0 - f_iso_f[j]) * hot
                P_prob /= P_prob.sum()
                P_fk[start + j, :] = _finish(P_prob)                # sum=NPIX or 1

        meta = dict(
            normalize=normalize, f0=f0, alpha=alpha_M, beta=beta_M,
            f_iso0=float(f_iso), f_iso_gamma=f_iso_gamma,
            kappa_min=kappa_min, kappa_max=kappa_max,
            used_callables=dict(
                amplitudes_fn=amplitudes_fn is not None,
                kappa_fn     =kappa_fn      is not None,
                f_iso_fn     =f_iso_fn      is not None,
                drift_fn     =drift_fn      is not None,
            )
        )
        return P_fk, theta, phi, w0, kappa0, meta
    


