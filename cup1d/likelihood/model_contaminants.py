"""Container for contaminant nuisance models."""

from __future__ import annotations

from typing import Any

from cup1d.contaminants import (
    hcd_boss,
    hcd_model_McDonald2005,
    hcd_model_rogers_class,
    si_add,
    si_mult,
    si_vid_final,
)


class Contaminants:
    """Bundle metal, HCD, and optional feedback contaminant models.

    Parameters
    ----------
    free_param_names : list[str], optional
        List of free parameter names.
    metal_models : dict, optional
        Dictionary of pre-initialized metal models.
    hcd_model : Any, optional
        Pre-initialized HCD model.
    sn_model : Any, optional
        Pre-initialized SN model.
    agn_model : Any, optional
        Pre-initialized AGN model.
    pars_cont : dict, optional
        Dictionary of contaminant parameters.
    ic_correction : Any, optional
        Initial condition correction.

    Attributes
    ----------
    pars_cont : dict
        Contaminant parameters.
    ic_correction : Any
        IC correction.
    metal_models : dict
        Dictionary of metal models.
    hcd_model : Any
        HCD model.
    sn_model : Any
        SN model.
    agn_model : Any
        AGN model.
    """

    def __init__(
        self,
        free_param_names: list[str] | None = None,
        metal_models: dict | None = None,
        hcd_model: Any | None = None,
        sn_model: Any | None = None,
        agn_model: Any | None = None,
        pars_cont: dict | None = None,
        ic_correction: Any | None = None,
    ):
        """Build contaminant models from a parameter dictionary."""
        if pars_cont is None:
            pars_cont = {}
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

        # if joint_model:
        self.metal_add = [
            # "CIVa_CIVb",
            # "MgIIa_MgIIb",
            "Si_add",
        ]

        # setup metal models
        self.metal_models = {}
        key = "Si_mult"
        try:
            self.metal_models[key] = metal_models[key]
        except (TypeError, KeyError):
            if pars_cont.get("metal_model_type") == "SiVid":
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
        try:
            self.metal_models[key] = metal_models[key]
        except (TypeError, KeyError):
            self.metal_models[key] = si_add.SiAdd(
                free_param_names=free_param_names,
                fid_vals=fid_vals,
                prop_coeffs=prop_coeffs,
                z_max=z_max,
                flat_priors=flat_priors,
                Gauss_priors=Gauss_priors,
            )

        # setup HCD model
        if hcd_model is not None:
            self.hcd_model = hcd_model
        else:
            hcd_model_type = pars_cont.get("hcd_model_type")
            if hcd_model_type == "McDonald":
                self.hcd_model = hcd_model_McDonald2005.HCDModel(
                    free_param_names=free_param_names,
                    fid_vals=fid_vals,
                    prop_coeffs=prop_coeffs,
                    z_max=z_max,
                    flat_priors=flat_priors,
                    Gauss_priors=Gauss_priors,
                )
            elif hcd_model_type == "boss":
                self.hcd_model = hcd_boss.HCDModel(
                    free_param_names=free_param_names,
                    fid_vals=fid_vals,
                    prop_coeffs=prop_coeffs,
                    z_max=z_max,
                    flat_priors=flat_priors,
                    Gauss_priors=Gauss_priors,
                )
            elif hcd_model_type == "new_rogers":
                self.hcd_model = hcd_model_rogers_class.HCDModel(
                    free_param_names=free_param_names,
                    fid_vals=fid_vals,
                    prop_coeffs=prop_coeffs,
                    z_max=z_max,
                    flat_priors=flat_priors,
                    Gauss_priors=Gauss_priors,
                )
            else:
                self.hcd_model = None

        self.sn_model = sn_model
        self.agn_model = agn_model

    def get_parameters(self) -> list[str]:
        """Return list of free parameter names from all models.

        Returns
        -------
        list[str]
            List of free parameter names.
        """
        params = []
        for model in self.metal_models:
            for par in self.metal_models[model].get_parameters():
                params.append(par)

        if self.hcd_model is not None:
            for par in self.hcd_model.get_parameters():
                params.append(par)

        if self.sn_model is not None:
            for par in self.sn_model.get_parameters():
                params.append(par)

        if self.agn_model is not None:
            for par in self.agn_model.get_parameters():
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
        for model in self.metal_models:
            if pname in self.metal_models[model].get_parameters():
                return self.metal_models[model].get_parameter(pname)

        if self.hcd_model is not None:
            if pname in self.hcd_model.get_parameters():
                return self.hcd_model.get_parameter(pname)

        if self.sn_model is not None:
            if pname in self.sn_model.get_parameters():
                return self.sn_model.get_parameter(pname)

        if self.agn_model is not None:
            if pname in self.agn_model.get_parameters():
                return self.agn_model.get_parameter(pname)

        raise ValueError(f"Parameter not found: {pname}")
