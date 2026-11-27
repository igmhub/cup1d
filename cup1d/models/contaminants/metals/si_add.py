import numpy as np
from cup1d.models.contaminants.base_contaminants import Contaminant
from cup1d.utils.batch import as_redshift_rows, unbatch
from cup1d.models.contaminants.metals._common import rstrength, vel_diff


class SiAdd(Contaminant):
    """Additive Si II cross-correlation contribution to the flux P1D.

    Redshift-history coefficients control additive metal-power templates in
    velocity-space units.
    """

    def __init__(
        self,
        coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        fid_vals=None,
        null_vals=None,
        flat_priors=None,
        z_max=None,
        Gauss_priors=None,
    ):
        """Initialize Si II transition ratios and redshift coefficient histories.

        Parameters are forwarded to :class:`Contaminant`; amplitude and
        damping histories are pivot polynomials by default.
        """

        self.wav = {
            # "SiIII": 1206.50,
            "SiIIc": 1260.42,
            "SiIIb": 1193.28,
            "SiIIa": 1190.42,
        }
        self.osc_strength = {
            "SiIIc": 1.22,
            "SiIIb": 0.575,
            "SiIIa": 0.277,
        }

        self.dv = {
            # "SiIII_SiIIc": vel_diff(self.wav["SiIII"], self.wav["SiIIc"]),
            # "SiIII_SiIIb": vel_diff(self.wav["SiIII"], self.wav["SiIIb"]),
            # "SiIII_SiIIa": vel_diff(self.wav["SiIII"], self.wav["SiIIa"]),
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
            "SiIIa_SiIIb": rstrength(
                self.wav["SiIIa"],
                self.wav["SiIIb"],
                self.osc_strength["SiIIa"],
                self.osc_strength["SiIIb"],
            ),
        }

        self.off = {
            "SiIIc_SiIIb": 0,
            "SiIIc_SiIIa": 0,
            "SiIIb_SiIIa": 1,
            "SiIIacbc": 0,
            "SiIIacab": 0,
            "SiIIbcab": 0,
        }

        list_coeffs = [
            "f_SiIIa_SiIIb",
            "s_SiIIa_SiIIb",
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
        """
        off = dict(self.off)
        if remove is not None:
            off.update({key: value for key, value in remove.items() if key in off})
        return off

    def get_contamination(self, z, k_kms, mF=None, like_params=None, remove=None):
        """Additive Si II x Si II P1D on velocity k grids.

        Parameters
        ----------
        z : array-like
            Redshifts.
        k_kms : sequence of ndarray
            Per-redshift wavenumbers in s/km.
        mF : array-like, optional
            Unused (interface of the metal models).
        like_params : mapping, optional
            Scalar values (one point) or columns of shape ``(n_batch,)``.
        remove : mapping, optional
            Pair flags overriding selected transition contributions.

        Returns
        -------
        list of ndarray
            One array per redshift: ``(n_k,)`` for one point,
            ``(n_batch, n_k)`` for a batch.
        """
        z, k_kms = as_redshift_rows(z, k_kms)
        values, scalar = self.histories(z, like_params)
        off = self.switches(remove)
        rac, rbc, rab = self.rat["SiIIa_SiIIc"], self.rat["SiIIb_SiIIc"], self.rat["SiIIa_SiIIb"]
        dv = self.dv
        # (flag, first and second velocity offsets, coefficients of the
        # difference and sum terms) of the three cross pairs
        pairs = (
            ("SiIIacbc", "SiIIc_SiIIa", "SiIIc_SiIIb", 2 * (1 + rac * rbc), 2 * (rac + rbc)),
            ("SiIIacab", "SiIIc_SiIIa", "SiIIb_SiIIa", 2 * rbc * (1 + rac * rab), 2 * rbc * (rac + rab)),
            ("SiIIbcab", "SiIIc_SiIIb", "SiIIb_SiIIa", 2 * rbc * (1 + rbc * rab), 2 * rbc * (rbc + rab)),
        )
        output = []
        for iz in range(len(z)):
            k = k_kms[iz][None, :]
            amp = values["f_SiIIa_SiIIb"][:, iz, None]
            damping = np.exp(-values["s_SiIIa_SiIIb"][:, iz, None] ** 2 * k**2)
            total = (
                off["SiIIc_SiIIa"] * (1 + rac**2 + 2 * rac * np.cos(dv["SiIIc_SiIIa"] * k))
                + off["SiIIc_SiIIb"] * (1 + rbc**2 + 2 * rbc * np.cos(dv["SiIIc_SiIIb"] * k))
                + off["SiIIb_SiIIa"] * rbc**2 * (1 + rab**2 + 2 * rab * np.cos(dv["SiIIb_SiIIa"] * k))
            )
            for flag, first, second, coef_d, coef_s in pairs:
                half_diff = 0.5 * (dv[first] - dv[second])
                half_sum = 0.5 * (dv[first] + dv[second])
                total = total + off[flag] * (coef_d * np.cos(half_diff * k) + coef_s * np.cos(half_sum * k))
            output.append(amp**2 * total * damping)
        return unbatch(output, scalar)

