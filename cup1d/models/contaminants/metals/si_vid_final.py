import numpy as np
from cup1d.models.contaminants.base_contaminants import Contaminant
from cup1d.utils.batch import as_redshift_rows, unbatch
from cup1d.models.contaminants.metals._common import rstrength, vel_diff


class SiVid(Contaminant):
    """Ma et al. silicon velocity-distortion multiplicative P1D model.

    It represents line-pair distortions as a redshift-dependent factor on the
    modeled P1D.
    """

    def __init__(
        self,
        coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        fid_vals=None,
        null_vals=None,
        z_max=None,
        flat_priors=None,
        Gauss_priors=None,
    ):
        """Initialize SiVid transition data and pivot coefficient histories.

        Parameters are forwarded to :class:`Contaminant`; default amplitude and
        damping histories are exponentially transformed pivot polynomials.
        """

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
            "SiIIa_Lya": 0,
            "SiIIb_Lya": 0,
            "SiIIc_Lya": 0,
            "SiIII_SiIIa": 0,
            "SiIII_SiIIb": 0,
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
            # "f_SiIIa_SiIII",
            # "f_SiIIb_SiIII",
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

    def get_contamination(self, z, k_kms, mF, like_params=None, remove=None):
        """SiVid multiplicative correction (Ma+2026, Eq. 18) on velocity k grids.

        Parameters
        ----------
        z : array-like
            Redshifts.
        k_kms : sequence of ndarray
            Per-redshift wavenumbers in s/km.
        mF : array-like
            Mean flux, ``(n_z,)`` for one point or ``(n_batch, n_z)``.
        like_params : mapping, optional
            Scalar values (one point) or columns of shape ``(n_batch,)``.
        remove : mapping, optional
            Unused: only Lya x SiIII is modelled.

        Returns
        -------
        list of ndarray
            One array per redshift: ``(n_k,)`` for one point,
            ``(n_batch, n_k)`` for a batch.
        """
        z, k_kms = as_redshift_rows(z, k_kms)
        values, scalar = self.histories(z, like_params)
        mF = np.atleast_2d(np.asarray(mF, dtype=float))
        output = []
        for iz in range(len(z)):
            k = k_kms[iz][None, :]
            one_minus_flux = 1 - mF[:, iz, None]
            p_siiii = values["f_Lya_SiIII"][:, iz, None] / one_minus_flux * np.exp(
                values["s_Lya_SiIII"][:, iz, None] * k)
            p_lya_siiii = (values["f_Lya_SiII"][:, iz, None] / one_minus_flux
                           * np.cos(self.dv["SiIII_Lya"] * k)
                           * np.exp(-values["s_Lya_SiII"][:, iz, None] * k))
            output.append(1 + p_siiii + p_lya_siiii)
        return unbatch(output, scalar)

