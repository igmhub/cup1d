"""Multiplicative silicon-metal contamination model."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cup1d.contaminants.base_contaminants import Contaminant


def vel_diff(lambda1: float, lambda2: float) -> float:
    """Return the velocity separation between two rest wavelengths in km/s.

    Parameters
    ----------
    lambda1 : float
        First wavelength.
    lambda2 : float
        Second wavelength.

    Returns
    -------
    float
        Velocity separation in km/s.
    """
    c_kms = 299792.458
    return float(np.abs(np.log(lambda2 / lambda1)) * c_kms)


def rstrength(lambda1: float, lambda2: float, f1: float, f2: float) -> float:
    """Return the optically thin relative line strength.

    Parameters
    ----------
    lambda1 : float
        First wavelength.
    lambda2 : float
        Second wavelength.
    f1 : float
        First oscillator strength.
    f2 : float
        Second oscillator strength.

    Returns
    -------
    float
        Relative line strength.
    """
    return (lambda1 * f1) / (lambda2 * f2)


class SiMult(Contaminant):
    """Multiplicative SiII/SiIII correction for Lyman-alpha correlations.

    The default model evolves SiIII-Lya, SiII-Lya, and SiII-SiIII
    amplitudes as pivot polynomials around ``z_0``.

    Parameters
    ----------
    coeffs : dict | None
        Coefficients for the silicon correction.
    prop_coeffs : dict | None
        Properties of the coefficients.
    free_param_names : list[str] | None
        Names of the free parameters.
    z_0 : float, optional
        Pivot redshift. Default is 3.0.
    fid_vals : dict | None
        Fiducial values for the coefficients.
    null_vals : dict | None
        Null values for the coefficients.
    z_max : dict | None
        Maximum redshift for each coefficient.
    flat_priors : dict | None
        Flat priors for the coefficients.
    Gauss_priors : dict | None
        Gaussian priors for the coefficients.

    Attributes
    ----------
    wav : dict
        Rest wavelengths for silicon and Lyman-alpha lines.
    osc_strength : dict
        Oscillator strengths for silicon lines.
    dv : dict
        Velocity separations between lines.
    rat : dict
        Relative line strengths.
    off : dict
        Switches for different line-pair contributions.
    """

    def __init__(
        self,
        coeffs: dict | None = None,
        prop_coeffs: dict | None = None,
        free_param_names: list[str] | None = None,
        z_0: float = 3.0,
        fid_vals: dict | None = None,
        null_vals: dict | None = None,
        z_max: dict | None = None,
        flat_priors: dict | None = None,
        Gauss_priors: dict | None = None,
    ):
        """Build the multiplicative silicon correction."""

        self.wav = {
            "SiIII": 1206.51,
            "SiIIc": 1260.42,
            "SiIIb": 1193.28,
            "SiIIa": 1190.42,
            "Lya": 1215.67,
        }
        self.osc_strength = {
            "SiIII": 1.67,
            "SiIIc": 1.22,
            "SiIIb": 0.575,
            "SiIIa": 0.277,
        }

        self.dv = {
            "SiIII_Lya": vel_diff(self.wav["SiIII"], self.wav["Lya"]),
            "SiIIc_Lya": vel_diff(self.wav["SiIIc"], self.wav["Lya"]),
            "SiIIb_Lya": vel_diff(self.wav["SiIIb"], self.wav["Lya"]),
            "SiIIa_Lya": vel_diff(self.wav["SiIIa"], self.wav["Lya"]),
            "SiIII_SiIIc": vel_diff(self.wav["SiIII"], self.wav["SiIIc"]),
            "SiIII_SiIIb": vel_diff(self.wav["SiIII"], self.wav["SiIIb"]),
            "SiIII_SiIIa": vel_diff(self.wav["SiIII"], self.wav["SiIIa"]),
            "SiIIc_SiIIb": vel_diff(self.wav["SiIIc"], self.wav["SiIIb"]),
            "SiIIc_SiIIa": vel_diff(self.wav["SiIIc"], self.wav["SiIIa"]),
            "SiIIb_SiIIa": vel_diff(self.wav["SiIIb"], self.wav["SiIIa"]),
        }

        self.rat = {
            "SiIIa_SiIII": rstrength(
                self.wav["SiIIa"],
                self.wav["SiIII"],
                self.osc_strength["SiIIa"],
                self.osc_strength["SiIII"],
            ),
            "SiIIb_SiIII": rstrength(
                self.wav["SiIIb"],
                self.wav["SiIII"],
                self.osc_strength["SiIIb"],
                self.osc_strength["SiIII"],
            ),
            "SiIIc_SiIII": rstrength(
                self.wav["SiIIc"],
                self.wav["SiIII"],
                self.osc_strength["SiIIc"],
                self.osc_strength["SiIII"],
            ),
            "SiIIa_SiIIc": rstrength(
                self.wav["SiIIa"],
                self.wav["SiIIc"],
                self.osc_strength["SiIIa"],
                self.osc_strength["SiIIc"],
            ),
            "SiIIb_SiIIc": rstrength(
                self.wav["SiIIb"],
                self.wav["SiIIc"],
                self.osc_strength["SiIIb"],
                self.osc_strength["SiIIc"],
            ),
        }

        self.off = {
            "SiIII_Lya": 1,
            "SiIIa_Lya": 1,
            "SiIIb_Lya": 1,
            "SiIIc_Lya": 0,
            "SiIII_SiIIa": 1,
            "SiIII_SiIIb": 1,
            "SiIII_SiIIc": 0,
            "SiIIc_SiIIb": 0,
            "SiIIc_SiIIa": 0,
            "SiIIb_SiIIa": 0,
        }

        list_coeffs = [
            "f_Lya_SiIII",
            "s_Lya_SiIII",
            "f_Lya_SiII",
            "s_Lya_SiII",
            "f_SiIIa_SiIII",
            "f_SiIIb_SiIII",
            # "f_SiIIa_SiIIb",
        ]

        if flat_priors is None:
            flat_priors = {}
            for coeff in list_coeffs:
                if coeff.startswith("f"):
                    flat_priors[coeff] = [[-3, 3], [-11, 2]]
                else:
                    flat_priors[coeff] = [[-1, 1], [-10, 7]]

        if prop_coeffs is None:
            prop_coeffs = {}
            for coeff in list_coeffs:
                prop_coeffs[coeff + "_ztype"] = "pivot"
                prop_coeffs[coeff + "_otype"] = "exp"

        if fid_vals is None:
            fid_vals = {}
            for coeff in list_coeffs:
                fid_vals[coeff] = [0, -20.0]

        if null_vals is None:
            null_vals = {}
            for coeff in list_coeffs:
                null_vals[coeff] = -20.0

        if z_max is None:
            z_max = {}
            for coeff in list_coeffs:
                z_max[coeff] = 10

        super().__init__(
            coeffs=coeffs,
            list_coeffs=list_coeffs,
            prop_coeffs=prop_coeffs,
            free_param_names=free_param_names,
            z_0=z_0,
            fid_vals=fid_vals,
            null_vals=null_vals,
            z_max=z_max,
            flat_priors=flat_priors,
            Gauss_priors=Gauss_priors,
        )

    def get_contamination(
        self,
        z: npt.NDArray[np.float64],
        k_kms: list[npt.NDArray[np.float64]],
        mF: npt.NDArray[np.float64],
        like_params: list | None = None,
        remove: dict | None = None,
    ) -> list[npt.NDArray[np.float64]]:
        """Return the multiplicative silicon correction for each redshift bin.

        Parameters
        ----------
        z : npt.NDArray[np.float64]
            Redshift values, one per entry of ``k_kms``.
        k_kms : list[npt.NDArray[np.float64]]
            Wavenumber arrays in s/km.
        mF : npt.NDArray[np.float64]
            Mean transmitted flux values used to normalize metal amplitudes.
        like_params : list | None, optional
            Likelihood parameters used to override the fiducial coefficients.
            Default is None.
        remove : dict | None, optional
            Per-term switches for enabling or disabling individual line-pair
            contributions. Default is None.

        Returns
        -------
        list[npt.NDArray[np.float64]]
            Multiplicative silicon correction.
        """

        vals = {}
        for key in self.list_coeffs:
            vals[key] = np.atleast_1d(
                self.get_value(key, z, like_params=like_params)
            )
            if key in self.null_vals:
                if self.prop_coeffs[key + "_otype"] == "const":
                    null = self.null_vals[key]
                else:
                    null = np.exp(self.null_vals[key])
                _ = vals[key] <= null
                vals[key][_] = 0

        ra3 = self.rat["SiIIa_SiIII"]
        rb3 = self.rat["SiIIb_SiIII"]
        rc3 = self.rat["SiIIc_SiIII"]

        self.off = {
            "SiIII_Lya": 1,
            "SiIIa_Lya": 1,
            "SiIIb_Lya": 1,
            "SiIIc_Lya": 0,
            "SiIII_SiIIa": 1,
            "SiIII_SiIIb": 1,
            "SiIII_SiIIc": 0,
            "SiIIc_SiIIb": 0,
            "SiIIc_SiIIa": 0,
            "SiIIb_SiIIa": 0,
        }

        if remove is not None:
            for key in remove:
                if key in self.off:
                    self.off[key] = remove[key]

        # SiII-SiII only additive
        self.off["SiIIb_SiIIa"] = 0
        self.off["SiIIc_SiIIa"] = 0
        self.off["SiIIc_SiIIb"] = 0

        metal_corr = []

        for iz in range(len(z)):
            # k-dependent damping of Lya-SiIII
            G_SiIII_Lya = 2 - 2 / (
                1 + np.exp(-vals["s_Lya_SiIII"][iz] * k_kms[iz])
            )
            # k-dependent damping of Lya-SiII
            G_SiII_Lya = 2 - 2 / (
                1 + np.exp(-vals["s_Lya_SiII"][iz] * k_kms[iz])
            )

            # scale amplitude of Cmm
            if "f_SiIIb_SiIII" in vals:
                G_SiII_SiIII = vals["f_SiIIb_SiIII"][iz]
            else:
                G_SiII_SiIII = 1

            # deviations from optically-thin limit
            if "f_SiIIa_SiIII" in vals:
                f_SiIIa_SiIII = vals["f_SiIIa_SiIII"][iz]
            else:
                f_SiIIa_SiIII = 1

            # not relevant anymore modeled here anymore
            if "s_SiIIa_SiIIb" in vals:
                G_SiII_SiII = 2 - 2 / (
                    1 + np.exp(-vals["s_SiIIa_SiIIb"][iz] * k_kms[iz])
                )
            else:
                G_SiII_SiII = 1
            if "f_SiIIa_SiIIb" in vals:
                G_SiII_SiII *= vals["f_SiIIa_SiIIb"][iz]

            # amplitude of SiIII
            aSiIII = vals["f_Lya_SiIII"][iz] / (1 - mF[iz])
            # amplitude of SiII, rb3 for convenience
            aSiII = rb3 * vals["f_Lya_SiII"][iz] / (1 - mF[iz])
            # deviations of ra3 from optically-thin limit
            _ra3 = ra3 * f_SiIIa_SiIII

            C0 = aSiIII**2 * self.off["SiIII_Lya"] + aSiII**2 * (
                (_ra3 / rb3) ** 2 * self.off["SiIIa_Lya"]
                + self.off["SiIIb_Lya"]
                + (rc3 / rb3) ** 2 * self.off["SiIIc_Lya"]
            )

            CSiIII_Lya = (
                2
                * aSiIII
                * G_SiIII_Lya
                * self.off["SiIII_Lya"]
                * np.cos(self.dv["SiIII_Lya"] * k_kms[iz])
            )

            CSiII_Lya = (
                2
                * aSiII
                * G_SiII_Lya
                * (
                    self.off["SiIIa_Lya"]
                    * (_ra3 / rb3)
                    * np.cos(self.dv["SiIIa_Lya"] * k_kms[iz])
                    + self.off["SiIIb_Lya"]
                    * np.cos(self.dv["SiIIb_Lya"] * k_kms[iz])
                    + self.off["SiIIc_Lya"]
                    * (rc3 / rb3)
                    * np.cos(self.dv["SiIIc_Lya"] * k_kms[iz])
                )
            )

            Cam = CSiIII_Lya + CSiII_Lya

            Cmm = (
                2
                * aSiIII
                * aSiII
                * G_SiII_SiIII
                * (
                    self.off["SiIII_SiIIc"]
                    * (rc3 / rb3)
                    * np.cos(self.dv["SiIII_SiIIc"] * k_kms[iz])
                    + self.off["SiIII_SiIIb"]
                    * np.cos(self.dv["SiIII_SiIIb"] * k_kms[iz])
                    + self.off["SiIII_SiIIa"]
                    * (_ra3 / rb3)
                    * np.cos(self.dv["SiIII_SiIIa"] * k_kms[iz])
                )
            )

            Cm = (
                2
                * aSiII**2
                * G_SiII_SiII
                * (
                    self.off["SiIIc_SiIIb"]
                    * (rc3 / rb3)
                    * np.cos(self.dv["SiIIc_SiIIb"] * k_kms[iz])
                    + self.off["SiIIc_SiIIa"]
                    * (rc3 / rb3)
                    * (ra3 / rb3)
                    * np.cos(self.dv["SiIIc_SiIIa"] * k_kms[iz])
                    + self.off["SiIIb_SiIIa"]
                    * (_ra3 / rb3)
                    * np.cos(self.dv["SiIIb_SiIIa"] * k_kms[iz])
                )
            )

            metal_corr.append(1 + C0 + Cam + Cmm + Cm)

        return metal_corr
