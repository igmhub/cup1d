import numpy as np

from cup1d.models.contaminants.HCD import (
    hcd_model_rogers_class,
    hcd_boss,
)

from cup1d.models.contaminants.metals import (
    si_mult,
    si_vid_final,
    si_add,
    si_kaiser,
)
from cup1d.utils.batch import as_columns, as_redshift_rows, unbatch


class Contaminants(object):
    """Compose metal, HCD, and initial-condition P1D contamination models."""

    def __init__(
        self,
        free_param_names=None,
        metal_models=None,
        hcd_model=None,
        pars_cont=None,
        ic_correction=None,
    ):
        """Build native contaminant components from configuration or overrides.

        Parameters
        ----------
        free_param_names : sequence of str, optional
            Free contaminant coefficient names.
        metal_models, hcd_model : mapping or object, optional
            Prebuilt components replacing native construction.
        pars_cont : mapping
            Native contaminant coefficient histories, priors, and model types.
        ic_correction : bool, optional
            Enable the fitted Nyx initial-condition correction.
        """
        self.pars_cont = pars_cont
        self.ic_correction = ic_correction

        if "flat_priors" in pars_cont:
            flat_priors = pars_cont["flat_priors"]
        else:
            flat_priors = None
        if "Gauss_priors" in pars_cont:
            Gauss_priors = pars_cont["Gauss_priors"]
        else:
            Gauss_priors = None

        if "z_max" in pars_cont:
            z_max = pars_cont["z_max"]
        else:
            z_max = None

        prop_coeffs = {}
        fid_vals = {}
        for key in pars_cont:
            fid_vals[key] = pars_cont[key]
            for key2 in ["otype", "ztype", "znodes"]:
                key3 = key + "_" + key2
                if key3 in pars_cont:
                    prop_coeffs[key3] = pars_cont[key3]
                else:
                    if key3.endswith("otype"):
                        if key3.startswith("HCD_const"):
                            prop_coeffs[key3] = "const"
                        else:
                            prop_coeffs[key3] = "exp"
                    elif key3.endswith("ztype"):
                        prop_coeffs[key3] = "pivot"

        # metal models that add to the P1D (the others multiply it)
        self.metal_add = ["Si_add"]

        # setup metal models
        self.metal_models = {}
        if pars_cont.get("metal_model_type") == "Kaiser" and not metal_models:
            # Line-of-sight Kaiser silicon model, additive; replaces the
            # Si_mult and Si_add silicon models. Needs metal moments.
            self.metal_add = ["Si_kaiser"]
            self.metal_models["Si_kaiser"] = si_kaiser.SiKaiser(
                free_param_names=free_param_names,
                fid_vals=fid_vals,
                prop_coeffs=prop_coeffs,
                flat_priors=flat_priors,
                Gauss_priors=Gauss_priors,
            )
        if "Si_kaiser" not in self.metal_models:
            key = "Si_mult"
            if metal_models and key in metal_models:
                self.metal_models[key] = metal_models[key]
            else:
                if pars_cont["metal_model_type"] == "SiVid":
                    # Ma+2025 2509.08613
                    self.metal_models[key] = si_vid_final.SiVid(
                        free_param_names=free_param_names,
                        fid_vals=fid_vals,
                        prop_coeffs=prop_coeffs,
                        z_max=z_max,
                        flat_priors=flat_priors,
                        Gauss_priors=Gauss_priors,
                    )
                else:
                    self.metal_models[key] = si_mult.SiMult(
                        free_param_names=free_param_names,
                        fid_vals=fid_vals,
                        prop_coeffs=prop_coeffs,
                        z_max=z_max,
                        flat_priors=flat_priors,
                        Gauss_priors=Gauss_priors,
                    )

            key = "Si_add"
            if metal_models and key in metal_models:
                self.metal_models[key] = metal_models[key]
            else:
                self.metal_models[key] = si_add.SiAdd(
                    free_param_names=free_param_names,
                    fid_vals=fid_vals,
                    prop_coeffs=prop_coeffs,
                    z_max=z_max,
                    flat_priors=flat_priors,
                    Gauss_priors=Gauss_priors,
                )

        # setup HCD model
        if hcd_model:
            self.hcd_model = hcd_model
        else:
            if pars_cont["hcd_model_type"] == "new_rogers":
                self.hcd_model = hcd_model_rogers_class.HCD_Model_Rogers(
                    free_param_names=free_param_names,
                    z_0=pars_cont.get("hcd_z_0", 3.0),
                    fid_vals=fid_vals,
                    prop_coeffs=prop_coeffs,
                    flat_priors=flat_priors,
                    Gauss_priors=Gauss_priors,
                )
            elif pars_cont["hcd_model_type"] == "BOSS":
                self.hcd_model = hcd_boss.HCD_BOSS(
                    free_param_names=free_param_names,
                    fid_vals=fid_vals,
                    prop_coeffs=prop_coeffs,
                    flat_priors=flat_priors,
                    Gauss_priors=Gauss_priors,
                )
            else:
                raise ValueError(
                    "hcd_model_type must be 'new_rogers' (Rogers+18 templates) or 'BOSS'"
                )

    def get_contamination(self, z, k_kms, mF, M_of_z=None, like_params=None, remove=None,
                          metal_moments=None):
        """All contaminant factors for one parameter point or a batch.

        Parameters
        ----------
        z : array-like
            Redshifts.
        k_kms : sequence of ndarray
            Per-redshift wavenumbers in s/km.
        mF : array-like
            Mean flux, ``(n_z,)`` for one point or ``(n_batch, n_z)``.
        M_of_z : array-like, optional
            Unused (kept for the feedback models, which are not active).
        like_params : mapping, optional
            Scalar values (one point) or columns of shape ``(n_batch,)``.
        remove : mapping, optional
            Metal transition flags overriding the default contributions (for
            figures; the models are not changed).
        metal_moments : sequence of mapping, optional
            Per-redshift metal moments, required by the Kaiser metal model.

        Returns
        -------
        dict of str to list of ndarray
            ``cont_mul_metals``, ``cont_HCD`` and ``IC_corr`` (dimensionless)
            and ``cont_add_metals`` (P1D units): one array per redshift,
            ``(n_k,)`` for one point and ``(n_batch, n_k)`` for a batch. The
            observed P1D is ``C_HCD C_mul IC P + C_add`` (then the resolution
            correction; :func:`cup1d.models.observation.apply_observation_model`).
        """
        z, k_kms = as_redshift_rows(z, k_kms)
        _, n_batch, scalar = as_columns(like_params)
        shapes = [(n_batch, len(k)) for k in k_kms]
        mult = [np.ones(shape) for shape in shapes]
        add = [np.zeros(shape) for shape in shapes]
        for name, model in self.metal_models.items():
            extra = {"metal_moments": metal_moments} if name == "Si_kaiser" else {}
            correction = model.get_contamination(
                z, k_kms, mF, like_params=like_params, remove=remove, **extra)
            for iz in range(len(z)):
                if name in self.metal_add:
                    add[iz] = add[iz] + correction[iz]
                else:
                    mult[iz] = mult[iz] * correction[iz]
        hcd = self.hcd_model.get_contamination(z, k_kms, like_params=like_params)
        hcd = [np.broadcast_to(hcd[iz], shapes[iz]) for iz in range(len(z))]
        if self.ic_correction:
            ic = [np.broadcast_to(row, shapes[iz])
                  for iz, row in enumerate(ref_nyx_ic_correction(k_kms, z))]
        else:
            ic = [np.ones(shape) for shape in shapes]
        out = {"cont_mul_metals": mult, "cont_add_metals": add, "cont_HCD": hcd, "IC_corr": ic}
        return {key: unbatch(rows, scalar) for key, rows in out.items()}

def ref_nyx_ic_correction(k_kms, z):
    # This is the function fitted from the comparison of two Nyx runs,
    # one with 2lpt (single fluid) IC and the other one with monofonic (2 fluid)
    # - The high k points and z evolution are well determined
    # - Low k term: quite uncertain, due to cosmic variance
    """Return the fitted Nyx initial-condition correction to P1D.

    Parameters
    ----------
    k_kms : numpy.ndarray or sequence of numpy.ndarray
        Wavenumbers in s/km, one (possibly ragged) grid per redshift; one
        array for a single redshift.
    z : array_like
        Redshifts.

    Returns
    -------
    list of numpy.ndarray
        Dimensionless multiplicative correction, one array per redshift.

    Notes
    -----
    The fit compares single-fluid 2LPT and two-fluid initial conditions.
    Its polynomial redshift dependence multiplies
    ``1-exp(-k_kms/0.003669741766936781)`` in percent; the correction is
    the reciprocal of one minus that fractional difference.
    """
    ic_corr_z = np.array([0.15261529, -2.30600644, 2.61877894])
    ic_corr_k = 0.003669741766936781
    z, k_kms = as_redshift_rows(z, k_kms)
    corrections = []
    for zz, k in zip(z, k_kms):
        percent = np.polyval(ic_corr_z, zz) * (1 - np.exp(-k / ic_corr_k))
        corrections.append(1 / (1 - percent / 100))
    return corrections
