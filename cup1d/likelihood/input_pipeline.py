"""YAML-based analysis arguments."""

import os
from copy import deepcopy
from pathlib import Path

import numpy as np


_TRAINING_SETS = {
    "CH24_mpgcen_gpr": "Cabayol23",
    "CH24_nyxcen_gpr": "models_Nyx_Sept2025_include_Nyx_fid_rseed",
}


def get_training_set(emulator_label):
    """Return the simulation archive associated with an emulator."""

    return _TRAINING_SETS.get(
        emulator_label,
        "Cabayol23" if "mpg" in emulator_label else "Pedersen21",
    )


class Args:
    """Analysis arguments resolved from CM2026 defaults and YAML overrides."""

    _CONT_PARAMS = {
        "f_Lya_SiIII": [0, -20.0], "s_Lya_SiIII": [0, 2.1],
        "f_Lya_SiII": [0, -20.0], "s_Lya_SiII": [0, 2.1],
        "f_SiIIa_SiIIb": [0, -20.0], "s_SiIIa_SiIIb": [0, 0.1],
        "f_SiIIa_SiIII": [0, 0.0], "f_SiIIb_SiIII": [0, 0.0],
        "HCD_damp1": [0, -20.0], "HCD_damp2": [0, -20.0],
        "HCD_damp3": [0, -20.0], "HCD_damp4": [0, -20.0],
        "HCD_const": [0, 0.0],
    }
    _SYST_PARAMS = {"R_coeff": [0, 0.0]}
    _CONT_FLAT_PRIORS = {
        "f_Lya_SiIII": [[-1, 1], [-6, -2]],
        "s_Lya_SiIII": [[-1, 1], [2, 7]],
        "f_Lya_SiII": [[-1, 1], [-6, -2]],
        "s_Lya_SiII": [[-1, 1], [2, 7]],
        "f_SiIIa_SiIIb": [[-1, 4], [-3, 3]],
        "s_SiIIa_SiIIb": [[-1, 3], [0, 7.5]],
        "f_SiIIa_SiIII": [[-1, 2], [-1, 3]],
        "f_SiIIb_SiIII": [[-1, 1], [-1, 5]],
        "HCD_damp1": [[-0.5, 0.5], [-10.0, -0.03]],
        "HCD_damp2": [[-0.5, 0.5], [-10.0, -1.0]],
        "HCD_damp3": [[-0.5, 0.5], [-10.0, -1.0]],
        "HCD_damp4": [[-0.5, 0.5], [-10.0, -1.0]],
        "HCD_const": [[-1, 1], [-0.2, 0.2]],
    }
    _FIDUCIAL_VALUES = {
        "fid_igm": {"tau_eff": 0.0, "sigT_kms": 1.0, "gamma": 1.0, "kF_kms": 1.0},
        "fid_cont": {
            "f_Lya_SiIII": -4.0, "s_Lya_SiIII": 5.0,
            "f_Lya_SiII": -4.0, "s_Lya_SiII": 5.5,
            "f_SiIIa_SiIIb": 0.5, "s_SiIIa_SiIIb": 4.0,
            "f_SiIIa_SiIII": 1.0, "f_SiIIb_SiIII": 1.0,
            "HCD_damp1": -1.4, "HCD_damp2": -6.0,
            "HCD_damp3": -5.0, "HCD_damp4": -5.0, "HCD_const": 0.0,
        },
        "fid_syst": {"R_coeff": 0.0},
    }
    _NULL_VALUES = {
        "tau_eff": 0.0, "sigT_kms": 1.0, "gamma": 1.0, "kF_kms": 1.0,
        "f_Lya_SiIII": -10.0, "s_Lya_SiIII": 2.1,
        "f_Lya_SiII": -10.0, "s_Lya_SiII": 2.1,
        "f_SiIIa_SiIIb": -10.0, "s_SiIIa_SiIIb": 0.1,
        "f_SiIIa_SiIII": 0.0, "f_SiIIb_SiIII": 0.0,
        "HCD_damp1": -10.0, "HCD_damp2": -10.0,
        "HCD_damp3": -10.0, "HCD_damp4": -10.0,
        "HCD_const": 0.0, "R_coeff": 0.0,
    }

    def __init__(self, synthetic=False, **options):
        from cup1d.config import apply_overrides, restore_runtime_types
        from cup1d.configuration import (
            make_cm2026_defaults,
            make_cm2026_synth_defaults,
            update_cm2026_derived,
        )

        factory = make_cm2026_synth_defaults if synthetic else make_cm2026_defaults
        config = apply_overrides(factory(), options, verbose=False)
        config = update_cm2026_derived(config, options)
        config = restore_runtime_types(config)
        for name, value in config.items():
            setattr(self, name, value)
        if self.file_ic is not None and not os.path.isabs(self.file_ic):
            self.file_ic = os.path.join(self.path_ic, self.file_ic)
        self.training_set = get_training_set(self.emulator_label)
        self.cont_params = {name: values.copy() for name, values in self._CONT_PARAMS.items()}
        self.syst_params = {name: values.copy() for name, values in self._SYST_PARAMS.items()}
        self._set_covariance_redshifts()
        self._set_parameter_nodes()
        self._set_fiducial_values()
        self._set_contaminant_priors()

    def _set_parameter_nodes(self):
        """Construct model redshift nodes from the analysis settings."""

        for section, names in ((self.fid_igm, self.igm_params), (self.fid_cont, self.cont_params), (self.fid_syst, self.syst_params)):
            for name in names:
                n_nodes = section.get(f"n_{name}", 0)
                key = f"{name}_znodes"
                if n_nodes > 0 and key not in section:
                    if n_nodes == 1:
                        section[key] = np.asarray([self.z_star])
                    elif section is self.fid_syst:
                        section[key] = np.linspace(self.z_min, self.z_max, n_nodes)
                    else:
                        section[key] = np.geomspace(self.z_min, self.z_max, n_nodes)

    def _set_covariance_redshifts(self):
        """Construct the covariance grid and expand its scalar factors."""

        self.cov_factor["z"] = np.arange(self.z_min, self.z_max + 0.5 * self.zbin_width, self.zbin_width)
        n_redshifts = len(self.cov_factor["z"])
        for name in ("val_stat", "val_syst", "val_emu", "val_full"):
            value = self.cov_factor[name]
            if np.isscalar(value):
                self.cov_factor[name] = np.full(n_redshifts, value)

    def _set_fiducial_values(self):
        """Create internal model reference values from the parameter layout."""

        sections = {"fid_igm": self.igm_params, "fid_cont": self.cont_params, "fid_syst": self.syst_params}
        for section, names in sections.items():
            values = getattr(self, section)
            for name in names:
                n_nodes = values.get(f"n_{name}", 0)
                reference = self._FIDUCIAL_VALUES[section][name] if n_nodes > 0 else self._NULL_VALUES[name]
                if values.get(f"{name}_ztype") == "pivot":
                    values[name] = [0, reference]
                else:
                    values[name] = np.full(len(values.get(f"{name}_znodes", [])), reference)

    def _set_contaminant_priors(self):
        """Set built-in flat priors and ensure they contain reference values."""

        priors = deepcopy(self._CONT_FLAT_PRIORS)
        variation = getattr(self, "name_variation", None)
        if variation is not None and variation.startswith("sim_"):
            for name in ("f_Lya_SiIII", "f_Lya_SiII", "f_SiIIa_SiIIb", "HCD_damp1", "HCD_damp2", "HCD_damp3", "HCD_damp4"):
                priors[name][-1][0] = -10.5
        for name, bounds in priors.items():
            reference = self.fid_cont[name][-1]
            if reference < bounds[-1][0]:
                bounds[-1][0] = reference - 0.1
            if reference > bounds[-1][1]:
                bounds[-1][1] = reference + 0.1
        self.fid_cont["flat_priors"] = priors

    @classmethod
    def from_yaml(cls, filename, verbose=True, synthetic=False, **options):
        """Create arguments from CM2026 defaults and a YAML override file."""

        from cup1d.config import read_config

        overrides = read_config(filename)
        overrides = _merge_overrides(overrides, options)
        return cls._from_overrides(
            overrides, verbose=verbose, synthetic=synthetic
        )

    @classmethod
    def from_baseline(cls, verbose=False, synthetic=False, **options):
        """Create arguments for the CM2026 baseline analysis."""

        return cls.from_yaml(
            _cm2026_config_dir() / "cm2026_base.yaml",
            verbose=verbose,
            synthetic=synthetic,
            **options,
        )

    @classmethod
    def _from_overrides(cls, overrides, verbose=True, synthetic=False):
        """Resolve an override mapping against the appropriate defaults."""

        from cup1d.config import (
            apply_overrides,
            print_resolved_values,
            restore_runtime_types,
        )
        from cup1d.configuration import (
            make_cm2026_defaults,
            make_cm2026_synth_defaults,
            update_cm2026_derived,
        )

        factory = make_cm2026_synth_defaults if synthetic else make_cm2026_defaults
        defaults = factory()
        config = apply_overrides(defaults, overrides, verbose=False)
        config = update_cm2026_derived(config, overrides)
        config = restore_runtime_types(config)
        if verbose:
            print_resolved_values(config, overrides)
        return cls(synthetic=synthetic, **config)

    @classmethod
    def from_variation(cls, name, verbose=False, synthetic=False, **options):
        """Apply a named CM2026 variation on top of the baseline."""

        if name in (None, "None"):
            return cls.from_baseline(
                verbose=verbose, synthetic=synthetic, **options
            )

        from cup1d.config import read_config

        config_dir = _cm2026_config_dir()
        overrides = _merge_overrides(
            read_config(config_dir / "cm2026_base.yaml"),
            read_config(config_dir / "variations" / f"{name}.yaml"),
        )
        overrides = _merge_overrides(overrides, options)
        return cls._from_overrides(
            overrides, verbose=verbose, synthetic=synthetic
        )


def _merge_overrides(base, updates):
    """Recursively combine YAML and programmatic overrides."""

    merged = deepcopy(base)
    for name, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(name), dict):
            merged[name] = _merge_overrides(merged[name], value)
        else:
            merged[name] = value
    return merged


def _cm2026_config_dir():
    """Return the directory containing the CM2026 YAML inputs."""

    from cup1d.utils.utils import get_path_repo

    return Path(get_path_repo("cup1d")) / "configs" / "cm2026"
