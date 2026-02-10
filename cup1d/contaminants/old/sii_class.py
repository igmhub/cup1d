import numpy as np
from cup1d.nuisance.base_contaminants import Contaminant


def vel_diff(lambda1, lambda2):
    c_kms = 299792.458
    return np.abs(np.log(lambda2 / lambda1)) * c_kms


def rstrength(lambda1, lambda2, f1, f2):
    return (lambda1 * f1) / (lambda2 * f2)


class SiiModel(Contaminant):
    """Model the contamination from Silicon Lya cross-correlations"""

    def __init__(
        self,
        coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        fid_vals=None,
        flat_priors=None,
        Gauss_priors=None,
    ):
        """Model the evolution of a metal contamination (SiII or SiIII).
        We use a power law around z_0=3."""

        self.wav = {
            "SiIII": 1206.50,
            "SiIIc": 1260.42,
            "SiIIb": 1193.28,
            "SiIIa": 1190.42,
            "Lya": 1215.67,
        }
        self.osc_strength = {
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

        list_coeffs = [
            "f_Lya_SiIII",
            "f_Lya_SiII",
            "s_Lya_SiIII",
            "s_Lya_SiII",
            "s_SiII_SiIII",
        ]

        if flat_priors is None:
            flat_priors = {
                "f_Lya_SiIII": [[-3, 3], [-11, -1]],
                "f_Lya_SiII": [[-3, 3], [-11, -1]],
                "s_Lya_SiIII": [[-1, 1], [-10, 7]],
                "s_Lya_SiII": [[-1, 1], [-10, 7]],
                "s_SiII_SiIII": [[-1, 1], [-10, 7]],
            }

        if prop_coeffs is None:
            prop_coeffs = {
                "f_Lya_SiIII" + "_ztype": "pivot",
                "f_Lya_SiII" + "_ztype": "pivot",
                "s_Lya_SiIII" + "_ztype": "pivot",
                "s_Lya_SiII" + "_ztype": "pivot",
                "s_SiII_SiIII" + "_ztype": "pivot",
                "f_Lya_SiIII" + "_otype": "exp",
                "f_Lya_SiII" + "_otype": "exp",
                "s_Lya_SiIII" + "_otype": "exp",
                "s_Lya_SiII" + "_otype": "exp",
                "s_SiII_SiIII" + "_otype": "exp",
            }

        if fid_vals is None:
            fid_vals = {
                "f_Lya_SiIII": [0, -10.5],
                "f_Lya_SiII": [0, -10.5],
                "s_Lya_SiIII": [0, -10.5],
                "s_Lya_SiII": [0, -10.5],
                "s_SiII_SiIII": [0, -10.5],
            }

        super().__init__(
            coeffs=coeffs,
            list_coeffs=list_coeffs,
            prop_coeffs=prop_coeffs,
            free_param_names=free_param_names,
            z_0=z_0,
            fid_vals=fid_vals,
            flat_priors=flat_priors,
            Gauss_priors=Gauss_priors,
        )

    def get_contamination(self, z, k_kms, mF, like_params=[], remove=None):
        """Multiplicative contamination at a given z and k (in s/km).
        The mean flux (mF) is used scale it (see McDonald et al. 2006)"""

        vals = {}
        for key in self.list_coeffs:
            vals[key] = np.atleast_1d(
                self.get_value(key, z, like_params=like_params)
            )

        r1 = self.rat["SiIIa_SiIIc"]
        r2 = self.rat["SiIIb_SiIIc"]

        off = {
            "SiIII_Lya": 1,
            "SiIIa_Lya": 1,
            "SiIIb_Lya": 1,
            "SiIIc_Lya": 1,
            "SiIII_SiIIa": 1,
            "SiIII_SiIIb": 1,
            "SiIII_SiIIc": 1,
            "SiIIc_SiIIb": 0,
            "SiIIc_SiIIa": 0,
            "SiIIb_SiIIa": 1,
        }
        if remove is not None:
            for key in remove:
                off[key] = remove[key]

        metal_corr = []

        for iz in range(len(z)):
            aSiIII = vals["f_Lya_SiIII"][iz] / (1 - mF[iz])
            aSiII = vals["f_Lya_SiII"][iz] / (1 - mF[iz])

            G_SiIII_Lya = 2 - 2 / (
                1 + np.exp(-vals["s_Lya_SiIII"][iz] * k_kms[iz])
            )

            G_SiII_Lya = 2 - 2 / (
                1 + np.exp(-vals["s_Lya_SiII"][iz] * k_kms[iz])
            )
            doppler = np.exp(
                -1 * vals["s_SiII_SiIII"][iz] ** 2 * k_kms[iz] ** 2
            )

            G_SiII_SiIII = 1

            C0 = aSiIII**2 * off["SiIII_Lya"] + aSiII**2 * (
                1 * off["SiIIc_Lya"]
                + r1**2 * off["SiIIa_Lya"]
                + r2**2 * off["SiIIb_Lya"]
            )

            CSiIII_Lya = (
                2
                * aSiIII
                * off["SiIII_Lya"]
                * np.cos(self.dv["SiIII_Lya"] * k_kms[iz])
            )

            CSiII_Lya = (
                2
                * aSiII
                * (
                    off["SiIIc_Lya"] * np.cos(self.dv["SiIIc_Lya"] * k_kms[iz])
                    + off["SiIIb_Lya"]
                    * r2
                    * np.cos(self.dv["SiIIb_Lya"] * k_kms[iz])
                    + off["SiIIa_Lya"]
                    * r1
                    * np.cos(self.dv["SiIIa_Lya"] * k_kms[iz])
                )
            )

            Cam = CSiIII_Lya * G_SiIII_Lya + CSiII_Lya * G_SiII_Lya

            Cmm = (
                2
                * aSiIII
                * aSiII
                * G_SiII_SiIII
                * (
                    off["SiIII_SiIIc"]
                    * np.cos(self.dv["SiIII_SiIIc"] * k_kms[iz])
                    + off["SiIII_SiIIb"]
                    * r2
                    * np.cos(self.dv["SiIII_SiIIb"] * k_kms[iz])
                    + off["SiIII_SiIIa"]
                    * r1
                    * np.cos(self.dv["SiIII_SiIIa"] * k_kms[iz])
                )
            )

            Cm = (
                2
                * aSiII**2
                * (
                    off["SiIIc_SiIIb"]
                    * r2
                    * np.cos(self.dv["SiIIc_SiIIb"] * k_kms[iz])
                    + off["SiIIc_SiIIa"]
                    * r1
                    * np.cos(self.dv["SiIIc_SiIIa"] * k_kms[iz])
                    + off["SiIIb_SiIIa"]
                    * r1
                    * r2
                    * np.cos(self.dv["SiIIb_SiIIa"] * k_kms[iz])
                )
            )

            metal_corr.append(1 + C0 + Cam + Cmm + Cm * doppler)

        return metal_corr
