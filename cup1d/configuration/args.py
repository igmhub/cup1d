"""YAML-based analysis arguments."""

import os
from copy import deepcopy
from pathlib import Path

import numpy as np


_TRAINING_SETS = {
    "lace_mpg": "Cabayol23",
    "lace_nyx": "models_Nyx_Sept2025_include_Nyx_fid_rseed",
    "CH24_mpgcen_gpr": "Cabayol23",
    "CH24_nyxcen_gpr": "models_Nyx_Sept2025_include_Nyx_fid_rseed",
}


def get_training_set(emulator_label):
    """Return the training archive selected by an emulator label.

    Parameters
    ----------
    emulator_label : str
        Public cup1d alias or underlying emulator label. Known LaCE aliases
        map to their explicit archive; unknown labels use the ``"mpg"``
        substring heuristic.

    Returns
    -------
    str
        Archive label. Unknown MPG-like labels select ``"Cabayol23"`` and
        all other labels select ``"Pedersen21"``.
    """

    return _TRAINING_SETS.get(
        emulator_label,
        "Cabayol23" if "mpg" in emulator_label else "Pedersen21",
    )


class Args:
    """Analysis arguments resolved from CM2026 defaults and YAML overrides.

    Most analysis choices originate in YAML. Stable package-level choices,
    including broad cosmological prior limits, are internal defaults and can
    still be overridden with an optional YAML ``cosmo_priors`` section.
    """

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
    _COSMO_PRIORS = {
        "ombh2": [0.018, 0.026],
        "omch2": [0.10, 0.14],
        "As": None,
        "ns": None,
        "mnu": [0.0, 1.0],
        "nrun": None,
        "H0": [50.0, 100.0],
    }

    def __init__(self, synthetic=False, **options):
        """Resolve CM2026 defaults and programmatic overrides.

        Parameters
        ----------
        synthetic : bool, default=False
            Select synthetic-data rather than observational CM2026 defaults.
        **options
            Nested configuration overrides using the same keys as the CM2026
            YAML. Values are merged with defaults and normalized to runtime
            array types.

        Notes
        -----
        This constructor does not read a YAML file. Use :meth:`from_yaml` or
        :meth:`from_variation` for file-backed configurations.
        """
        from cup1d.configuration.loader import apply_overrides, restore_runtime_types
        from cup1d.configuration import (
            make_cm2026_defaults,
            make_cm2026_synth_defaults,
            update_cm2026_derived,
        )

        self.synthetic = synthetic
        factory = make_cm2026_synth_defaults if synthetic else make_cm2026_defaults
        defaults = factory()
        # Keep stable broad limits out of the baseline YAML while allowing a
        # dedicated YAML overlay to override individual entries.
        defaults["cosmo_priors"] = deepcopy(self._COSMO_PRIORS)
        config = apply_overrides(defaults, options, verbose=False)
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
                if name in values:
                    self._preserve_configured_fiducial_values(
                        values, name, n_nodes
                    )
                    continue
                reference = self._FIDUCIAL_VALUES[section][name] if n_nodes > 0 else self._NULL_VALUES[name]
                if values.get(f"{name}_ztype") == "pivot":
                    values[name] = [0, reference]
                else:
                    values[name] = np.full(len(values.get(f"{name}_znodes", [])), reference)

    @staticmethod
    def _preserve_configured_fiducial_values(values, name, n_nodes):
        """Normalize an explicit fiducial history to its configured layout.

        Parameters
        ----------
        values : dict
            Model configuration section mutated in place.
        name : str
            Physical history name whose value is normalized.
        n_nodes : int
            Number of configured redshift nodes for interpolated histories.

        Raises
        ------
        ValueError
            If a pivot history has more than one value, or an interpolated
            history has neither one nor ``n_nodes`` values.
        """

        configured = np.asarray(values[name])
        if values.get(f"{name}_ztype") == "pivot":
            if configured.size != 1:
                raise ValueError(
                    f"Pivot parameter {name} requires one configured value"
                )
            values[name] = [0, configured.item()]
        elif configured.size == 1:
            values[name] = np.full(n_nodes, configured.item())
        elif configured.size == n_nodes:
            values[name] = configured
        else:
            raise ValueError(
                f"Configured {name} has {configured.size} values for "
                f"{n_nodes} redshift nodes"
            )

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
        """Create resolved arguments from a CM2026-compatible YAML file.

        Parameters
        ----------
        filename : str or pathlib.Path
            YAML override file. Its resolved absolute path is retained as
            ``config_path`` on the returned object.
        verbose : bool, default=True
            Print resolved default and user-provided values.
        synthetic : bool, default=False
            Select synthetic-data rather than observational defaults.
        **options
            Programmatic overrides merged after YAML values.

        Returns
        -------
        Args
            Fully resolved analysis arguments with ``config_loader="yaml"``.
        """

        from cup1d.configuration.loader import read_config

        overrides = read_config(filename)
        overrides = _merge_overrides(overrides, options)
        args = cls._from_overrides(
            overrides, verbose=verbose, synthetic=synthetic
        )
        args.config_path = str(Path(filename).expanduser().resolve())
        args.config_loader = "yaml"
        return args

    @classmethod
    def from_baseline(cls, verbose=False, synthetic=False, **options):
        """Create arguments from the canonical CM2026 baseline YAML.

        Parameters
        ----------
        verbose : bool, default=False
            Print resolved default and override values.
        synthetic : bool, default=False
            Select the synthetic default family before applying overrides.
        **options
            Programmatic CM2026 overrides.

        Returns
        -------
        Args
            Resolved baseline analysis arguments.
        """

        return cls.from_yaml(
            _cm2026_config_dir() / "cm2026_base.yaml",
            verbose=verbose,
            synthetic=synthetic,
            **options,
        )

    @classmethod
    def _from_overrides(cls, overrides, verbose=True, synthetic=False):
        """Resolve a mapping against CM2026 defaults.

        Parameters
        ----------
        overrides : dict
            Nested configuration values to merge into the selected defaults.
        verbose : bool, default=True
            Print resolved leaves and their source.
        synthetic : bool, default=False
            Select synthetic-data rather than observational defaults.

        Returns
        -------
        Args
            Fully initialized arguments with derived nodes and priors.
        """

        from cup1d.configuration.loader import (
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
        defaults["cosmo_priors"] = deepcopy(cls._COSMO_PRIORS)
        config = apply_overrides(defaults, overrides, verbose=False)
        config = update_cm2026_derived(config, overrides)
        config = restore_runtime_types(config)
        if verbose:
            print_resolved_values(config, overrides)
        return cls(synthetic=synthetic, **config)

    @classmethod
    def from_variation(cls, name, verbose=False, synthetic=False, **options):
        """Create arguments by applying a CM2026 variation to the baseline.

        Parameters
        ----------
        name : str or pathlib.Path or None
            Variation name under ``configs/cm2026/variations`` or a YAML path.
            ``None`` and ``"None"`` select the baseline without a variation.
        verbose : bool, default=False
            Print resolved default and override values.
        synthetic : bool, default=False
            Select synthetic-data rather than observational defaults.
        **options
            Programmatic overrides merged after baseline and variation YAML.

        Returns
        -------
        Args
            Resolved arguments with ``config_loader="variation"`` unless no
            variation was requested.
        """

        if name in (None, "None"):
            return cls.from_baseline(
                verbose=verbose, synthetic=synthetic, **options
            )

        from cup1d.configuration.loader import read_config

        config_dir = _cm2026_config_dir()
        variation_path = Path(name).expanduser()
        if variation_path.suffix not in {".yaml", ".yml"}:
            variation_path = config_dir / "variations" / f"{name}.yaml"
        else:
            variation_path = variation_path.resolve()
        overrides = _merge_overrides(
            read_config(config_dir / "cm2026_base.yaml"),
            read_config(variation_path),
        )
        overrides = _merge_overrides(overrides, options)
        args = cls._from_overrides(
            overrides, verbose=verbose, synthetic=synthetic
        )
        args.config_path = str(variation_path.resolve())
        args.config_loader = "variation"
        return args


def _merge_overrides(base, updates):
    """Recursively merge nested override mappings.

    Parameters
    ----------
    base, updates : dict
        Base mapping and values that replace or recursively extend it.

    Returns
    -------
    dict
        Deep copy of ``base`` with ``updates`` applied. Inputs are unchanged.
    """

    merged = deepcopy(base)
    for name, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(name), dict):
            merged[name] = _merge_overrides(merged[name], value)
        else:
            merged[name] = value
    return merged


def _cm2026_config_dir():
    """Return the installed repository directory containing CM2026 YAML files.

    Returns
    -------
    pathlib.Path
        ``configs/cm2026`` under the cup1d repository root.
    """

    from cup1d.utils.utils import get_path_repo

    return Path(get_path_repo("cup1d")) / "configs" / "cm2026"
