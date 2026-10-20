"""Silicon contamination of P1D from a line-of-sight Kaiser 3D model.

Each silicon transition ``m`` is a linear tracer of the matter field with flux
bias ``b_m`` and redshift-space parameter ``beta_m``, whose absorption is
displaced along the line of sight by its velocity offset from Lya. The 3D
power of a pair of components ``a, b`` is

    P_ab(k, mu) = b_a b_b (1 + beta_a mu^2)(1 + beta_b mu^2) P_L(k) D_ab W_ab,

with the Lya x Lya term given by ForestFlow. ``D_ab = d_a d_b`` per leg: the
Lya leg carries ``sqrt(D_NL)`` and a metal leg the gas pressure smoothing
``d_m = exp(-(k/k_p)^2 / 2)`` with ForestFlow's Arinyo ``k_p`` (metals trace
the same gas). ``W_am = exp(-(k_par sigma_m)^2 / 2)`` is a line-of-sight
decorrelation between Lya and metal ``m`` only; the metal autos and the
crosses between metal lines are not decorrelated (``W_mn = 1``). The same
model is used for the 3D correlation function in Vega.

Projecting to P1D, the line-of-sight displacement becomes ``cos(k dv)`` and,
because ``W_m`` depends only on ``k_par``, the metal parameters factor out of
the transverse integral:

    P1D_am = b_m W_m (A_0 + beta_m A_1),
    P1D_mn = b_m b_n (I_0 + (beta_m + beta_n) I_1 + beta_m beta_n I_2),

with the moments ``A_j`` and ``I_j`` of
:func:`cup1d.likelihood.external.project_metal_moments`. The contamination is
additive:

    P1D_obs = P1D_Lya + sum_m 2 cos(k dv_am) P1D_am + sum_m P1D_mm
              + sum_{m<n} 2 cos(k dv_mn) P1D_mn.
"""

import numpy as np

from cup1d.models.contaminants.base_contaminants import Contaminant

C_KMS = 299792.458
WAVELENGTHS = {
    "Lya": 1215.67,
    "SiIII": 1206.51,
    "SiIIa": 1190.42,
    "SiIIb": 1193.28,
    "SiIIc": 1260.42,
}
# Species of each transition: beta and sigma are shared within a species.
SPECIES = {"SiIII": "SiIII", "SiIIa": "SiII", "SiIIb": "SiII", "SiIIc": "SiII"}
# Coefficient families: one bias per transition, beta and sigma per species.
FAMILIES = ("b_SiIII", "b_SiIIa", "b_SiIIb", "b_SiIIc",
            "beta_SiIII", "beta_SiII", "sig_SiIII", "sig_SiII")


def velocity_offset(line1, line2):
    """Return the line-of-sight velocity offset between two transitions in km/s."""
    return C_KMS * abs(np.log(WAVELENGTHS[line1] / WAVELENGTHS[line2]))


class SiKaiser(Contaminant):
    """Additive Kaiser silicon contamination of P1D.

    Parameters
    ----------
    dv_max_kms : float, default=26000.0
        Pairs whose velocity offset exceeds this (about 250 Mpc/h at z=2.33)
        are not included. All Lya and silicon pairs lie below it.
    **kwargs
        Forwarded to :class:`Contaminant`: the redshift histories of every
        family (``n_<family>``, ``<family>_ztype``, ``<family>_znodes``,
        ``<family>_otype``), fiducial values and priors, as for every other
        cup1d contaminant.

    Notes
    -----
    Coefficient families: ``b_SiIII``, ``b_SiIIa`` (1190), ``b_SiIIb`` (1193),
    ``b_SiIIc`` (1260) give the magnitude ``|b|`` of each transition's flux
    bias (metal flux biases are negative; the sign is applied here). With
    ``otype: exp`` their coefficients are ``ln|b|`` and two nodes interpolated
    linearly give a power-law-like evolution, as the DR1 metal amplitudes.
    ``beta_SiIII``, ``beta_SiII`` and ``sig_SiIII``, ``sig_SiII`` (Gaussian
    line-of-sight decorrelation of Lya x metal, in km/s) are per species.
    """

    def __init__(self, dv_max_kms=26000.0, **kwargs):
        super().__init__(list_coeffs=list(FAMILIES), **kwargs)
        lines = list(SPECIES)
        self.cross_lya = [(m, velocity_offset("Lya", m)) for m in lines
                          if velocity_offset("Lya", m) <= dv_max_kms]
        self.cross_metal = [(m, n, velocity_offset(m, n))
                            for i, m in enumerate(lines) for n in lines[i + 1:]
                            if velocity_offset(m, n) <= dv_max_kms]
        self.lines = lines

    def coefficients(self, z, like_params=None):
        """Return per-transition ``b``, ``beta`` and ``sigma`` at redshift ``z``.

        Parameters
        ----------
        z : float
            Redshift.
        like_params : mapping, optional
            Coefficient overrides keyed by indexed parameter name.

        Returns
        -------
        dict
            ``{line: (bias, beta, sigma_kms)}``; biases are negative.
        """
        out = {}
        for line in self.lines:
            species = SPECIES[line]
            out[line] = (
                -float(self.get_value("b_" + line, z, like_params)),
                float(self.get_value("beta_" + species, z, like_params)),
                float(self.get_value("sig_" + species, z, like_params)),
            )
        return out

    def p1d_terms(self, z, k_kms, moments, like_params=None):
        """Return each additive metal term at one redshift.

        Parameters
        ----------
        z : float
            Redshift.
        k_kms : array_like
            Parallel wavenumbers in s/km.
        moments : mapping
            Moments ``A0, A1, I0, I1, I2`` on ``k_kms`` in km/s.
        like_params : mapping, optional
            Coefficient overrides.

        Returns
        -------
        dict
            Terms in km/s keyed ``"Lya_<line>"``, ``"<line>_<line>"`` and
            ``"<line1>_<line2>"``, each already including ``2 cos(k dv)`` for
            crosses. Their sum is the additive contamination.
        """
        k = np.asarray(k_kms, dtype=float)
        A0, A1 = (np.asarray(moments[name], dtype=float) for name in ("A0", "A1"))
        I0, I1, I2 = (np.asarray(moments[name], dtype=float) for name in ("I0", "I1", "I2"))
        par = self.coefficients(z, like_params)
        window = {line: np.exp(-0.5 * (k * sigma) ** 2) for line, (_, _, sigma) in par.items()}

        def metal_metal(m, n):
            # not decorrelated: the decorrelation acts only on Lya x metal
            bm, betam, _ = par[m]
            bn, betan, _ = par[n]
            return bm * bn * (I0 + (betam + betan) * I1 + betam * betan * I2)

        terms = {}
        for line, dv in self.cross_lya:
            bias, beta, _ = par[line]
            terms["Lya_" + line] = 2 * np.cos(k * dv) * bias * window[line] * (A0 + beta * A1)
        for line in self.lines:
            terms[line + "_" + line] = metal_metal(line, line)
        for m, n, dv in self.cross_metal:
            terms[m + "_" + n] = 2 * np.cos(k * dv) * metal_metal(m, n)
        return terms

    def get_contamination(self, z, k_kms, mF=None, like_params=None, remove=None,
                          metal_moments=None):
        """Return the additive metal contamination of P1D in km/s.

        Parameters
        ----------
        z : array_like
            Redshifts.
        k_kms : sequence of ndarray
            Per-redshift wavenumbers in s/km.
        mF : array_like, optional
            Unused; accepted for interface compatibility.
        like_params : mapping, optional
            Coefficient overrides.
        remove : mapping, optional
            Term names (as returned by :meth:`p1d_terms`) mapped to true to
            omit them.
        metal_moments : sequence of mapping
            Per-redshift moments on ``k_kms``.

        Returns
        -------
        list of ndarray
            Additive P1D terms in km/s, one per redshift.

        Raises
        ------
        ValueError
            If the moments are missing.
        """
        if metal_moments is None:
            raise ValueError(
                "the Kaiser metal model needs metal moments from a ForestFlow "
                "projection (cup1d.likelihood.external.project_metal_moments)")
        remove = {} if remove is None else remove
        out = []
        for iz, zz in enumerate(np.atleast_1d(z)):
            terms = self.p1d_terms(zz, k_kms[iz], metal_moments[iz], like_params)
            total = np.zeros(len(k_kms[iz]))
            for name, term in terms.items():
                if not remove.get(name, False):
                    total = total + term
            out.append(total)
        return out
