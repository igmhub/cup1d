"""Container for IGM nuisance models and fiducial histories."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from cup1d.igm.mean_flux_class import MeanFlux
from cup1d.igm.pressure_class import Pressure
from cup1d.igm.thermal_class import Thermal
from cup1d.utils.utils import get_path_repo, is_number_string


class IGM:
    """Bundle mean-flux, thermal, and pressure IGM models.

    Parameters
    ----------
    free_param_names : list[str], optional
        List of free parameter names.
    pars_igm : dict, optional
        Dictionary of IGM parameters.
    F_model : MeanFlux, optional
        Mean flux model.
    T_model : Thermal, optional
        Thermal model.
    P_model : Pressure, optional
        Pressure model.

    Attributes
    ----------
    fid_sim_igm_mF : str
        Fiducial simulation label for mean flux.
    fid_sim_igm_T : str
        Fiducial simulation label for thermal history.
    fid_sim_igm_kF : str
        Fiducial simulation label for pressure.
    priors : dict
        Prior bounds for IGM parameters.
    models : dict
        Dictionary of IGM models.
    fid_igm : dict
        Fiducial IGM history evaluated on a redshift grid.
    """

    def __init__(
        self,
        free_param_names: list[str] | None = None,
        pars_igm: dict | None = None,
        F_model: MeanFlux | None = None,
        T_model: Thermal | None = None,
        P_model: Pressure | None = None,
    ):
        """Build IGM models from a parameter dictionary."""
        if pars_igm is None:
            pars_igm = {}

        # set simulation from which we get fiducial IGM history
        for key in ["mF", "T", "kF"]:
            lab = "label_" + key
            if lab in pars_igm:
                setattr(self, "fid_sim_igm_" + key, pars_igm[lab])
            else:
                setattr(self, "fid_sim_igm_" + key, "mpg_central")

        if "Gauss_priors" in pars_igm:
            Gauss_priors = pars_igm["Gauss_priors"]
        else:
            Gauss_priors = None

        prop_coeffs = {}
        fid_vals = {}
        for key in ["tau_eff", "gamma", "sigT_kms", "kF_kms"]:
            if key in pars_igm:
                fid_vals[key] = pars_igm[key]
            for key2 in ["otype", "ztype", "znodes"]:
                key3 = key + "_" + key2
                if key3 in pars_igm:
                    prop_coeffs[key3] = pars_igm[key3]
                else:
                    if key3 == "tau_eff_otype":
                        prop_coeffs[key3] = "exp"
                    else:
                        if key3.endswith("otype"):
                            prop_coeffs[key3] = "const"
                        elif key3.endswith("ztype"):
                            prop_coeffs[key3] = "pivot"

        if "priors" in pars_igm:
            fact_priors = pars_igm["priors"]
        else:
            fact_priors = 1.0

        fid_igm = self.get_igm(
            sim_igm_mF=self.fid_sim_igm_mF,
            sim_igm_T=self.fid_sim_igm_T,
            sim_igm_kF=self.fid_sim_igm_kF,
        )

        self.set_priors(fid_igm, prop_coeffs, fact_priors=fact_priors)

        self.models = {
            "F_model": F_model,
            "T_model": T_model,
            "P_model": P_model,
        }

        for key in self.models:
            if self.models[key] is None:
                if key == "F_model":
                    model = MeanFlux
                elif key == "T_model":
                    model = Thermal
                elif key == "P_model":
                    model = Pressure

                self.models[key] = model(
                    free_param_names=free_param_names,
                    fid_igm=fid_igm,
                    fid_vals=fid_vals,
                    prop_coeffs=prop_coeffs,
                    flat_priors=self.priors,
                    Gauss_priors=Gauss_priors,
                )

    def set_fid_igm(self, zs: np.ndarray) -> None:
        """Evaluate fiducial IGM histories on redshift grid ``zs``.

        Parameters
        ----------
        zs : np.ndarray
            Redshift grid.
        """
        self.fid_igm = {}
        self.fid_igm["z"] = zs
        for key in self.models:
            for key2 in self.models[key].list_coeffs:
                if key2 == "tau_eff":
                    self.fid_igm[key] = self.models[key].get_tau_eff(zs)
                elif key2 == "gamma":
                    self.fid_igm[key] = self.models[key].get_gamma(zs)
                elif key2 == "sigT_kms":
                    self.fid_igm[key] = self.models[key].get_sigT_kms(zs)
                elif key2 == "kF_kms":
                    self.fid_igm[key] = self.models[key].get_kF_kms(zs)

    def get_igm(
        self,
        sim_igm_mF: str = "mpg_central",
        sim_igm_T: str = "mpg_central",
        sim_igm_kF: str = "mpg_central",
    ) -> dict[str, Any]:
        """Return IGM histories for specified simulation labels.

        Parameters
        ----------
        sim_igm_mF : str, optional
            Label for mean flux history. Default is 'mpg_central'.
        sim_igm_T : str, optional
            Label for thermal history. Default is 'mpg_central'.
        sim_igm_kF : str, optional
            Label for pressure history. Default is 'mpg_central'.

        Returns
        -------
        dict
            Dictionary of IGM histories.
        """
        igm = {}
        for key in ["mF", "T", "kF"]:
            if key == "mF":
                sim_igm = sim_igm_mF
            elif key == "T":
                sim_igm = sim_igm_T
            elif key == "kF":
                sim_igm = sim_igm_kF

            if sim_igm[:3] == "mpg":
                fname = os.path.join(
                    get_path_repo("lace"),
                    "data",
                    "sim_suites",
                    "Australia20",
                    "IGM_histories.npy",
                )
            elif sim_igm[:3] == "nyx":
                fname = os.path.join(
                    os.environ["NYX_PATH"],
                    "nyx_emu_IGM_models_Nyx_Mar2025_with_CGAN_val_3axes.npy",
                )

            try:
                data_igm = np.load(fname, allow_pickle=True).item()
            except Exception:
                raise ValueError(f"{fname} not found") from None

            if sim_igm in data_igm.keys():
                igm[key] = data_igm[sim_igm]
            else:
                raise ValueError(f"IGM not found in {fname} for {sim_igm}")

        return igm

    def set_priors(
        self, fid_igm: dict, prop_coeffs: dict, fact_priors: float = 1.0
    ) -> None:
        """Set prior bounds for IGM parameters based on fiducial histories.

        Parameters
        ----------
        fid_igm : dict
            Fiducial IGM histories.
        prop_coeffs : dict
            Properties of IGM parameters.
        fact_priors : float, optional
            Scaling factor for the priors. Default is 1.0.
        """
        self.priors = {}
        for key in ["tau_eff", "gamma", "sigT_kms", "kF_kms"]:
            if key == "tau_eff":
                mod = "mF"
            elif key in ["gamma", "sigT_kms"]:
                mod = "T"
            elif key == "kF_kms":
                mod = "kF"

            vals = fid_igm[mod][key]

            if prop_coeffs[key + "_otype"] == "exp":
                vals = np.log(vals)

            if prop_coeffs[key + "_ztype"] == "pivot":
                self.priors[key] = [
                    vals.min() - 0.5 * fact_priors,
                    vals.max() + 0.5 * fact_priors,
                ]
            elif prop_coeffs[key + "_ztype"].startswith("interp"):
                self.priors[key] = [
                    vals.min() - 0.5 * fact_priors,
                    vals.max() + 0.5 * fact_priors,
                ]

    def get_parameters(self) -> list[str]:
        """Return list of free parameter names from all models.

        Returns
        -------
        list[str]
            List of free parameter names.
        """
        params = []
        for model in self.models:
            for par in self.models[model].get_parameters():
                params.append(par)
        return params

    def get_parameter(self, pname: str) -> Any:
        """Return a likelihood parameter by name.

        Parameters
        ----------
        pname : str
            Parameter name.

        Returns
        -------
        Any
            Likelihood parameter object.
        """
        pname_orig, _ = is_number_string(pname)
        for model in self.models:
            if pname_orig in self.models[model].list_coeffs:
                return self.models[model].get_parameter(pname)
        raise ValueError("Parameter not found")
