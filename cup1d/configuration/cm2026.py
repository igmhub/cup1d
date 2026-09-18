"""Defaults and generic derivation rules for the CM2026 analysis."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cup1d.configuration.loader import restore_runtime_types
from cup1d.utils.utils import get_path_repo

_CONFIG_DIR = Path(get_path_repo("cup1d")) / "configs" / "cm2026"
_DEFAULTS_FILE = _CONFIG_DIR / "cm2026_defaults.yaml"
_SYNTH_DEFAULTS_FILE = _CONFIG_DIR / "cm2026_synth_defaults.yaml"
_CONFIG_GROUPS = (
    (
        "Data",
        (
            "data_label",
            "data_bias",
            "z_min",
            "z_max",
            "zbin_width",
            "k_rebin_factor",
            "p1d_fname",
            "path_data",
        ),
    ),
    (
        "Synthetic data",
        (
            "true_cosmo_label",
            "true_igm",
            "true_cont",
            "true_syst",
            "apply_smoothing",
            "synth_cov_label",
            "cov_label_hires",
            "add_noise",
            "seed_noise",
        ),
    ),
    ("Emulator", ("emulator_label", "drop_emu_sim")),
    (
        "Cosmology",
        ("fid_cosmo_label", "z_star", "kp_kms", "fix_cosmo", "vary_alphas"),
    ),
    ("IGM model", ("igm_params", "fid_igm")),
    ("Contamination model", ("fid_cont",)),
    ("Instrumental systematics", ("fid_syst", "ic_correction")),
    ("Covariance", ("cov_syst_type", "emu_cov_type", "cov_factor")),
    ("Priors", ("use_star_priors", "prior_Gauss_rms", "Gauss_priors")),
    (
        "Inference",
        (
            "fit_type",
            "mcmc",
            "initial_sampling_values",
            "file_ic",
            "path_ic",
        ),
    ),
    (
        "Output and runtime",
        (
            "out_folder",
            "path_out",
            "system",
            "verbose",
            "name_variation",
            "pre_defined",
        ),
    ),
)


def make_cm2026_defaults() -> dict[str, Any]:
    """Read the canonical observational CM2026 defaults."""

    return _read_defaults(_DEFAULTS_FILE)


def make_cm2026_synth_defaults() -> dict[str, Any]:
    """Read the canonical synthetic-data CM2026 defaults."""

    return _read_defaults(_SYNTH_DEFAULTS_FILE)


def _read_defaults(filename: Path) -> dict[str, Any]:
    """Load a user-facing defaults file and restore local runtime paths."""

    with filename.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    config = deepcopy(config)
    if not isinstance(config, dict):
        raise ValueError(f"CM2026 defaults in {filename} must be a mapping")
    config["path_ic"] = str(Path(get_path_repo("cup1d")) / "data" / "ics")
    return _organize_config(restore_runtime_types(config))


def update_cm2026_derived(
    config: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Update values derived from general redshift and model settings.

    Explicitly supplied derived values are preserved. This allows users to
    replace the standard node construction when needed.
    """

    z_min = config["z_min"]
    z_max = config["z_max"]

    parameter_sections = {
        "fid_igm": config["igm_params"],
        "fid_cont": _parameter_names(config["fid_cont"]),
        "fid_syst": _parameter_names(config["fid_syst"]),
    }
    for section, names in parameter_sections.items():
        values = config[section]
        for name in names:
            n_nodes = values.get(f"n_{name}", 0)
            nodes_key = f"{name}_znodes"
            if n_nodes > 0 and not _provided(overrides, section, nodes_key):
                if n_nodes == 1:
                    nodes = np.asarray([config["z_star"]])
                elif section == "fid_syst":
                    nodes = np.linspace(z_min, z_max, n_nodes)
                else:
                    nodes = np.geomspace(z_min, z_max, n_nodes)
                values[nodes_key] = nodes

    if not _provided(overrides, "cov_factor", "z"):
        width = config["zbin_width"]
        config["cov_factor"]["z"] = np.arange(
            z_min, z_max + 0.5 * width, width
        )

    n_cov = len(config["cov_factor"]["z"])
    for name in ("val_stat", "val_syst", "val_emu", "val_full"):
        value = config["cov_factor"][name]
        scalar = value[0] if isinstance(value, (list, np.ndarray)) else value
        config["cov_factor"][name] = np.full(n_cov, scalar)

    return config


def _provided(overrides: dict[str, Any], section: str, name: str) -> bool:
    section_values = overrides.get(section, {})
    return isinstance(section_values, dict) and name in section_values


def _parameter_names(section: dict[str, Any]) -> list[str]:
    """Infer model parameter names from their node-count settings."""

    return [name[2:] for name in section if name.startswith("n_")]


def _organize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Order top-level options by their role in the analysis."""

    organized = {}
    for _, names in _CONFIG_GROUPS:
        for name in names:
            if name in config:
                organized[name] = config[name]

    missing = config.keys() - organized.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"CM2026 options are missing a configuration group: {names}")
    return organized
