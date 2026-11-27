
from cup1d.models.contaminants import resolution


class Systematics(object):
    """Container for instrument-resolution systematic corrections.

    The object owns the resolution model and combines its correction with the
    likelihood's fitted parameter mapping.
    """

    def __init__(self, free_param_names=None, resolution_model=None, pars_syst=None):
        """Build a resolution model from native systematics configuration.

        Parameters
        ----------
        free_param_names : sequence of str, optional
            Free resolution coefficient names.
        resolution_model : object, optional
            Prebuilt model replacing native construction.
        pars_syst : mapping
            Fiducial coefficient histories, transforms, nodes, and priors.
        """
        self.pars_syst = pars_syst

        if "flat_priors" in pars_syst:
            flat_priors = pars_syst["flat_priors"]
        else:
            flat_priors = None
        if "Gauss_priors" in pars_syst:
            Gauss_priors = pars_syst["Gauss_priors"]
        else:
            Gauss_priors = None

        if "z_max" in pars_syst:
            z_max = pars_syst["z_max"]
        else:
            z_max = None

        prop_coeffs = {}
        fid_vals = {}
        for key in pars_syst:
            fid_vals[key] = pars_syst[key]
            for key2 in ["otype", "ztype", "znodes"]:
                key3 = key + "_" + key2
                if key3 in pars_syst:
                    prop_coeffs[key3] = pars_syst[key3]
                else:
                    if key3.endswith("otype"):
                        if key3.startswith("HCD_const"):
                            prop_coeffs[key3] = "const"
                        else:
                            prop_coeffs[key3] = "exp"
                    elif key3.endswith("ztype"):
                        prop_coeffs[key3] = "pivot"

        # setup Resolution model
        if resolution_model:
            self.resolution_model = resolution_model
        else:
            self.resolution_model = resolution.Resolution(
                free_param_names=free_param_names,
                form=pars_syst.get("R_form", "exponential"),
                fid_vals=fid_vals,
                prop_coeffs=prop_coeffs,
                flat_priors=flat_priors,
                Gauss_priors=Gauss_priors,
            )

    # def get_dict_cont(self):
    #     dict_out = {}

    #     if self.args.fid_syst["res_model_type"] == "pivot":
    #         for ii in range(len(self.args.fid_syst["R_coeff"])):
    #             dict_out["R_coeff_" + str(ii)] = self.args.fid_syst["R_coeff"][
    #                 -1 - ii
    #             ]
    #     else:
    #         for ii in range(self.resolution_model.get_Nparam()):
    #             dict_out["R_coeff_" + str(ii)] = 0

    #     return dict_out

    def get_contamination(self, z, k_kms, like_params=None):
        """Resolution correction for one point or a batch.

        Parameters
        ----------
        z : array-like
            Redshifts.
        k_kms : sequence of ndarray
            Per-redshift wavenumbers in s/km.
        like_params : mapping, optional
            Scalar values (one point) or columns of shape ``(n_batch,)``.

        Returns
        -------
        list of ndarray
            Dimensionless corrections, ``(n_k,)`` per redshift for one point
            or ``(n_batch, n_k)`` for a batch.
        """
        return self.resolution_model.get_contamination(z, k_kms, like_params=like_params)

