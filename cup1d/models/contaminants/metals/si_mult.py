import numpy as np
from cup1d.models.contaminants.base_contaminants import Contaminant
from cup1d.utils.batch import as_redshift_rows, unbatch
from cup1d.models.contaminants.metals._common import rstrength, vel_diff


class SiMult(Contaminant):
    """Multiplicative Si III/Si II cross-correlation P1D contamination model.

    The model applies redshift-dependent metal distortion factors to the
    uncontaminated Ly-alpha P1D.
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
        """Initialize silicon transition ratios and coefficient histories.

        Parameters are forwarded to :class:`Contaminant`; default histories
        are pivot polynomials with exponentially transformed amplitudes.
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

    def switches(self, remove=None):
        """Component switches for one evaluation.

        Parameters
        ----------
        remove : mapping, optional
            ``{component: 0 or 1}`` overrides of :attr:`off` (e.g. to remove
            a component in a figure). They apply to this evaluation only;
            :attr:`off` is never modified.

        Returns
        -------
        dict
            Switches; SiII--SiII terms are always off (they are additive,
            see :class:`SiAdd`).
        """
        off = dict(self.off)
        if remove is not None:
            off.update({key: value for key, value in remove.items() if key in off})
        off["SiIIb_SiIIa"] = off["SiIIc_SiIIa"] = off["SiIIc_SiIIb"] = 0
        return off

    def get_contamination(self, z, k_kms, mF, like_params=None, remove=None):
        """Multiplicative silicon correction on velocity k grids.

        Parameters
        ----------
        z : array-like
            Redshifts.
        k_kms : sequence of ndarray
            Per-redshift wavenumbers in s/km.
        mF : array-like
            Mean flux, ``(n_z,)`` for one point or ``(n_batch, n_z)``; it
            normalizes the metal amplitudes.
        like_params : mapping, optional
            Scalar values (one point) or columns of shape ``(n_batch,)``.
        remove : mapping, optional
            Pair flags overriding default transition contributions.

        Returns
        -------
        list of ndarray
            One array per redshift: ``(n_k,)`` for one point,
            ``(n_batch, n_k)`` for a batch.
        """
        z, k_kms = as_redshift_rows(z, k_kms)
        values, scalar = self.histories(z, like_params)
        mF = np.atleast_2d(np.asarray(mF, dtype=float))
        off = self.switches(remove)
        ra3, rb3, rc3 = self.rat["SiIIa_SiIII"], self.rat["SiIIb_SiIII"], self.rat["SiIIc_SiIII"]
        dv = self.dv
        output = []
        for iz in range(len(z)):
            k = k_kms[iz][None, :]

            def val(name, default=1.0):
                """Coefficient column at this redshift, or its neutral value."""
                return values[name][:, iz, None] if name in values else default

            def damping(name):
                """k-dependent damping ``2 - 2 / (1 + exp(-s k))`` (1 if absent)."""
                return 2 - 2 / (1 + np.exp(-values[name][:, iz, None] * k)) if name in values else 1.0

            one_minus_flux = 1 - mF[:, iz, None]
            a3 = val("f_Lya_SiIII") / one_minus_flux  # SiIII amplitude
            a2 = rb3 * val("f_Lya_SiII") / one_minus_flux  # SiII amplitude (rb3 for convenience)
            r3 = ra3 * val("f_SiIIa_SiIII")  # ra3 off the optically-thin limit
            c0 = a3**2 * off["SiIII_Lya"] + a2**2 * (
                (r3 / rb3) ** 2 * off["SiIIa_Lya"] + off["SiIIb_Lya"] + (rc3 / rb3) ** 2 * off["SiIIc_Lya"]
            )
            cam = 2 * a3 * damping("s_Lya_SiIII") * off["SiIII_Lya"] * np.cos(dv["SiIII_Lya"] * k)
            cam = cam + 2 * a2 * damping("s_Lya_SiII") * (
                off["SiIIa_Lya"] * (r3 / rb3) * np.cos(dv["SiIIa_Lya"] * k)
                + off["SiIIb_Lya"] * np.cos(dv["SiIIb_Lya"] * k)
                + off["SiIIc_Lya"] * (rc3 / rb3) * np.cos(dv["SiIIc_Lya"] * k)
            )
            cmm = 2 * a3 * a2 * val("f_SiIIb_SiIII") * (
                off["SiIII_SiIIc"] * (rc3 / rb3) * np.cos(dv["SiIII_SiIIc"] * k)
                + off["SiIII_SiIIb"] * np.cos(dv["SiIII_SiIIb"] * k)
                + off["SiIII_SiIIa"] * (r3 / rb3) * np.cos(dv["SiIII_SiIIa"] * k)
            )
            # SiII x SiII: switched off here (modelled additively by SiAdd)
            cm = 2 * a2**2 * damping("s_SiIIa_SiIIb") * val("f_SiIIa_SiIIb") * (
                off["SiIIc_SiIIb"] * (rc3 / rb3) * np.cos(dv["SiIIc_SiIIb"] * k)
                + off["SiIIc_SiIIa"] * (rc3 / rb3) * (ra3 / rb3) * np.cos(dv["SiIIc_SiIIa"] * k)
                + off["SiIIb_SiIIa"] * (r3 / rb3) * np.cos(dv["SiIIb_SiIIa"] * k)
            )
            output.append(1 + c0 + cam + cmm + cm)
        return unbatch(output, scalar)

