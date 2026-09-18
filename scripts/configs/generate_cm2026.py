"""Generate resolved input files for the Chaves-Montero et al. 2026 analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import numpy as np
import yaml

from cup1d.old_code.old_input import Args
from cup1d.configuration import make_cm2026_defaults, make_cm2026_synth_defaults
from cup1d.utils.utils import get_path_repo


REPO_ROOT = Path(get_path_repo("cup1d"))
CONFIG_DIR = REPO_ROOT / "configs" / "cm2026"
DEFAULTS_OUTPUT = CONFIG_DIR / "cm2026_defaults.yaml"
SYNTH_DEFAULTS_OUTPUT = CONFIG_DIR / "cm2026_synth_defaults.yaml"
BASELINE_OUTPUT = CONFIG_DIR / "cm2026_base.yaml"
VARIATIONS_DIR = CONFIG_DIR / "variations"

# File name -> inputs needed by the current Args configuration machinery.
# Most variations only need name_variation; the exceptional cases also change
# the dataset, emulator, or emulator covariance representation.
VARIATIONS: dict[str, dict[str, Any]] = {
    "zmin": {"name_variation": "zmin"},
    "zmax": {"name_variation": "zmax"},
    "DESIY1_QMLE": {"data_label": "DESIY1_QMLE"},
    "DESIY1_FFT3_dir": {"data_label": "DESIY1_FFT3_dir"},
    "data_syst_diag": {"name_variation": "data_syst_diag"},
    "no_inflate": {"name_variation": "no_inflate"},
    "no_emu_cov": {"name_variation": "no_emu_cov"},
    "emu_diag": {"emu_cov_type": "diagonal"},
    "emu_block": {"emu_cov_type": "block"},
    "infl_emu_cov": {"name_variation": "infl_emu_cov"},
    "bias_eBOSS": {"name_variation": "bias_eBOSS"},
    "nyx": {"emulator_label": "CH24_nyxcen_gpr"},
    "cosmo": {"name_variation": "cosmo"},
    "cosmo_74": {"name_variation": "cosmo_74"},
    "cosmo_mnu_varh": {"name_variation": "cosmo_mnu_varh"},
    "cosmo_high_3sig": {"name_variation": "cosmo_high_3sig"},
    "cosmo_low_3sig": {"name_variation": "cosmo_low_3sig"},
    "more_igm": {"name_variation": "more_igm"},
    "IGM_priors": {"name_variation": "IGM_priors"},
    "LLS_nz4": {"name_variation": "LLS_nz4"},
    "HCD0": {"name_variation": "HCD0"},
    "DLAs": {"name_variation": "DLAs"},
    "HCD_BOSS": {"name_variation": "HCD_BOSS"},
    "metal_thin": {"name_variation": "metal_thin"},
    "metal_deco": {"name_variation": "metal_deco"},
    "metal_si2": {"name_variation": "metal_si2"},
    "metal_trad": {"name_variation": "metal_trad"},
    "Metals_Ma2025": {"name_variation": "Metals_Ma2025"},
}

_UNCHANGED = object()

SECTION_HEADERS = {
    "data_label": "Data",
    "true_cosmo_label": "Synthetic data",
    "emulator_label": "Emulator",
    "fid_cosmo_label": "Cosmology",
    "igm_params": "IGM model",
    "fid_cont": "Contamination model",
    "fid_syst": "Instrumental systematics",
    "cov_syst_type": "Covariance",
    "use_star_priors": "Priors",
    "fit_type": "Inference",
    "out_folder": "Output and runtime",
}

_INTERNAL_FIDUCIAL_VALUES = {
    "fid_igm": {"tau_eff", "sigT_kms", "gamma", "kF_kms"},
    "fid_cont": {
        "f_Lya_SiIII", "s_Lya_SiIII", "f_Lya_SiII", "s_Lya_SiII",
        "f_SiIIa_SiIIb", "s_SiIIa_SiIIb", "f_SiIIa_SiIII",
        "f_SiIIb_SiIII", "HCD_damp1", "HCD_damp2", "HCD_damp3",
        "HCD_damp4", "HCD_const",
    },
    "fid_syst": {"R_coeff"},
}

TOP_LEVEL_COMMENTS = {
    "data_label": "P1D datasets included in the analysis.",
    "data_bias": "Multiplicative factor applied to the measured P1D.",
    "z_min": "Minimum redshift included in the analysis.",
    "z_max": "Maximum redshift included in the analysis.",
    "zbin_width": "Redshift-bin width used to construct regular grids.",
    "k_rebin_factor": "Factor controlling interpolation before k-bin rebinning.",
    "emulator_label": "Identifier of the P1D emulator.",
    "drop_emu_sim": "Simulation omitted from emulator training, or null to use all simulations.",
    "true_cosmo_label": "Cosmology used to generate synthetic data.",
    "fid_cosmo_label": "Fiducial cosmology used by the likelihood model.",
    "igm_params": "Names of the intergalactic-medium model parameters.",
    "cont_params": "Definitions and null values of contamination parameters.",
    "syst_params": "Definitions and null values of instrumental systematic parameters.",
    "true_igm": "IGM settings used to generate synthetic data.",
    "fid_igm": "Fiducial IGM model and its free-parameter configuration.",
    "true_cont": "Contamination settings used to generate synthetic data.",
    "fid_cont": "Fiducial contamination model and priors.",
    "true_syst": "Instrumental-systematic settings used to generate synthetic data.",
    "fid_syst": "Fiducial instrumental-systematic model.",
    "apply_smoothing": "Whether smoothing is applied when constructing mock P1D data.",
    "synth_cov_label": "Dataset used to construct synthetic-data covariance.",
    "cov_label_hires": "Dataset used for the high-resolution covariance model.",
    "cov_syst_type": "Structure assumed for the observational systematic covariance.",
    "z_star": "Pivot redshift for compressed linear-power parameters.",
    "kp_kms": "Pivot wavenumber in inverse velocity units.",
    "use_star_priors": "Optional priors on compressed linear-power parameters.",
    "add_noise": "Whether random noise is added to synthetic data.",
    "seed_noise": "Random seed used when adding noise.",
    "verbose": "Whether the analysis prints progress information.",
    "ic_correction": "Whether to apply the initial-condition correction.",
    "fix_cosmo": "Whether cosmological parameters are fixed.",
    "vary_alphas": "Whether running parameters of the primordial spectrum are varied.",
    "prior_Gauss_rms": "Optional common RMS for Gaussian priors.",
    "emu_cov_type": "Emulator covariance representation: full, diagonal, or block.",
    "mcmc": "Sampler configuration.",
    "out_folder": "Directory in which analysis products are written.",
    "Gauss_priors": "Optional parameter-specific Gaussian priors.",
    "system": "Optional computing-system identifier.",
    "path_out": "Base directory for analysis outputs.",
    "path_ic": "Directory containing initial-condition corrections.",
    "p1d_fname": "Optional explicit P1D measurement filename.",
    "pre_defined": "Name of the historical preset used to create this resolved file.",
    "file_ic": "Initial-condition filename, resolved relative to `path_ic`.",
    "path_data": "Optional base directory for input data.",
    "training_set": "Simulation archive used to train or construct the emulator.",
    "fit_type": "Inference configuration used for this analysis.",
    "name_variation": "Name of the analysis variation, or null for the baseline.",
    "cov_factor": "Redshift-dependent covariance rescaling factors.",
}

PARAMETER_COMMENTS = {
    "tau_eff": "Effective optical-depth evolution.",
    "sigT_kms": "Thermal broadening in km/s.",
    "gamma": "Slope of the IGM temperature-density relation.",
    "kF_kms": "Pressure-smoothing scale in inverse velocity units.",
    "f_Lya_SiIII": "Amplitude of the Lyman-alpha--SiIII metal contribution.",
    "s_Lya_SiIII": "Decorrelation scale of the Lyman-alpha--SiIII contribution.",
    "f_Lya_SiII": "Amplitude of the Lyman-alpha--SiII contribution.",
    "s_Lya_SiII": "Decorrelation scale of the Lyman-alpha--SiII contribution.",
    "f_SiIIa_SiIIb": "Amplitude of the SiIIa--SiIIb contribution.",
    "s_SiIIa_SiIIb": "Decorrelation scale of the SiIIa--SiIIb contribution.",
    "f_SiIIa_SiIII": "Relative amplitude of the SiIIa--SiIII contribution.",
    "f_SiIIb_SiIII": "Relative amplitude of the SiIIb--SiIII contribution.",
    "HCD_damp1": "Amplitude of the first high-column-density component.",
    "HCD_damp2": "Amplitude of the second high-column-density component.",
    "HCD_damp3": "Amplitude of the third high-column-density component.",
    "HCD_damp4": "Amplitude of the fourth high-column-density component.",
    "HCD_const": "Scale-independent high-column-density contribution.",
    "R_coeff": "Spectrograph-resolution correction coefficient.",
}


def to_builtin(value: Any) -> Any:
    """Convert values used by ``Args`` into YAML-safe Python objects."""

    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def make_baseline() -> dict[str, Any]:
    """Return the legacy fully resolved CM2026 baseline configuration."""

    args = Args(pre_defined="CM2026")
    return to_builtin(vars(args))


def make_variation(options: dict[str, Any]) -> dict[str, Any]:
    """Return the fully resolved configuration for one CM2026 variation."""

    data_label = options.get("data_label", "DESIY1_QMLE3")
    emulator_label = options.get("emulator_label", "CH24_mpgcen_gpr")
    emu_cov_type = options.get("emu_cov_type", "full")
    name_variation = options.get("name_variation")

    args = Args(
        data_label=[data_label],
        emulator_label=emulator_label,
        emu_cov_type=emu_cov_type,
    )
    args.set_baseline(
        fit_type="global_opt",
        fix_cosmo=False,
        name_variation=name_variation,
        mcmc_conf="explore",
    )
    return to_builtin(vars(args))


def config_diff(defaults: Any, variation: Any) -> Any:
    """Return the minimal recursive override from defaults to variation."""

    if isinstance(defaults, dict) and isinstance(variation, dict):
        changed = {}
        removed = defaults.keys() - variation.keys()
        if removed:
            changed["__delete__"] = sorted(removed)
        for key, value in variation.items():
            if key not in defaults:
                changed[key] = value
                continue
            difference = config_diff(defaults[key], value)
            if difference is not _UNCHANGED:
                changed[key] = difference
        return changed if changed else _UNCHANGED

    if defaults == variation:
        return _UNCHANGED
    return variation


def make_variation_overrides(name: str) -> dict[str, Any]:
    """Return minimal overrides for a legacy CM2026 variation."""

    baseline = make_baseline()
    variation = make_variation(VARIATIONS[name])
    for config in (baseline, variation):
        if config["file_ic"] is not None:
            config["file_ic"] = Path(config["file_ic"]).name
        config.pop("cont_params", None)
        config.pop("syst_params", None)
        config.pop("training_set", None)
        for parameter in config["igm_params"]:
            config["fid_igm"].pop(f"{parameter}_znodes", None)
        for parameter in _INTERNAL_FIDUCIAL_VALUES["fid_cont"]:
            config["fid_cont"].pop(f"{parameter}_znodes", None)
        config["fid_cont"].pop("flat_priors", None)
        for parameter in _INTERNAL_FIDUCIAL_VALUES["fid_syst"]:
            config["fid_syst"].pop(f"{parameter}_znodes", None)
        config["cov_factor"].pop("z", None)
        for name in ("val_stat", "val_syst", "val_emu", "val_full"):
            values = config["cov_factor"][name]
            if not all(value == values[0] for value in values):
                raise ValueError(f"Cannot collapse redshift-dependent cov_factor.{name}")
            config["cov_factor"][name] = values[0]
        for section, names in _INTERNAL_FIDUCIAL_VALUES.items():
            for parameter in names:
                config[section].pop(parameter, None)
    difference = config_diff(baseline, variation)
    return {} if difference is _UNCHANGED else difference


def add_yaml_comments(yaml_text: str) -> str:
    """Add a descriptive comment before every YAML mapping key."""

    output = []
    parents: list[tuple[int, str]] = []
    key_pattern = re.compile(r"^( *)([^\s][^:]*):(?:\s|$)")

    for line in yaml_text.splitlines():
        match = key_pattern.match(line)
        if match is None or line.lstrip().startswith("-"):
            output.append(line)
            continue

        indent = len(match.group(1))
        key = match.group(2).strip("'\"")
        while parents and parents[-1][0] >= indent:
            parents.pop()
        path = tuple(parent_key for _, parent_key in parents) + (key,)
        if indent == 0 and key in SECTION_HEADERS:
            output.append("")
            output.append(f"## {SECTION_HEADERS[key]}")
        output.append(f"{' ' * indent}# {_describe_yaml_key(path)}")
        output.append(line)
        parents.append((indent, key))

    return "\n".join(output) + "\n"


def _describe_yaml_key(path: tuple[str, ...]) -> str:
    """Return a human-readable description for one configuration key."""

    key = path[-1]
    if len(path) == 1:
        return TOP_LEVEL_COMMENTS[key]

    section = path[0]
    if section == "mcmc":
        return {
            "explore": "Whether walkers explore around the initial position.",
            "parallel": "Whether likelihood evaluations use parallel execution.",
            "n_burn_in": "Number of discarded burn-in steps per walker.",
            "n_steps": "Number of retained sampling steps per walker.",
            "n_walkers": "Number of ensemble-sampler walkers per free parameter.",
            "thin": "Keep one sample for every this many sampler steps.",
        }.get(key, f"Sampler setting `{key}`.")

    if section == "cov_factor":
        return {
            "z": "Redshifts at which covariance factors are defined.",
            "val_stat": "Multiplicative factors for statistical errors.",
            "val_syst": "Multiplicative factors for systematic errors.",
            "val_emu": "Multiplicative factors for emulator errors.",
            "val_full": "Multiplicative factors applied to the full covariance.",
        }.get(key, f"Covariance setting `{key}`.")

    if key == "flat_priors":
        return "Flat-prior bounds for contamination parameters."
    if len(path) >= 2 and path[-2] == "flat_priors":
        return f"Flat-prior bounds for {PARAMETER_COMMENTS.get(key, key)}"
    if key == "priors":
        return "Common scaling applied to the model priors."
    if key == "hcd_model_type":
        return "High-column-density absorption model."
    if key == "metal_model_type":
        return "Metal-contamination model."
    if key.startswith("label_"):
        return f"Simulation or observational reference for `{key[6:]}`."
    if key.startswith("n_"):
        parameter = key[2:]
        return f"Number of free redshift nodes for {PARAMETER_COMMENTS.get(parameter, parameter)}"
    if key.endswith("_znodes"):
        parameter = key[: -len("_znodes")]
        return f"Redshift-node positions for {PARAMETER_COMMENTS.get(parameter, parameter)}"
    if key.endswith("_ztype"):
        parameter = key[: -len("_ztype")]
        return f"Redshift interpolation scheme for {PARAMETER_COMMENTS.get(parameter, parameter)}"
    if key.endswith("_otype"):
        parameter = key[: -len("_otype")]
        return f"Coefficient parameterization for {PARAMETER_COMMENTS.get(parameter, parameter)}"
    if key.endswith("_fixed"):
        parameter = key[: -len("_fixed")]
        return f"Whether {PARAMETER_COMMENTS.get(parameter, parameter)} is fixed."
    if key in PARAMETER_COMMENTS:
        prefix = "True/mock" if section.startswith("true_") else "Fiducial"
        return f"{prefix} value(s): {PARAMETER_COMMENTS[key]}"

    return f"Resolved `{section}` setting `{key}`."


def write_yaml(
    config: dict[str, Any],
    output: Path,
    force: bool = False,
    description: str = "fully resolved configuration",
    comments: bool = False,
) -> None:
    """Write a configuration file, protecting existing files by default."""

    if output.exists() and not force:
        raise FileExistsError(
            f"{output} already exists; pass --force to overwrite it"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by scripts/configs/generate_cm2026.py.\n"
        f"# This is a {description} and can be edited directly.\n"
    )
    yaml_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if comments:
        yaml_text = add_yaml_comments(yaml_text)
    output.write_text(header + yaml_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="custom output path when generating one file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the output file if it already exists",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--variation",
        choices=sorted(VARIATIONS),
        help="generate one variation override",
    )
    selection.add_argument(
        "--all-variations",
        action="store_true",
        help="generate all variation override files",
    )
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    if options.all_variations:
        for name in VARIATIONS:
            output = VARIATIONS_DIR / f"{name}.yaml"
            write_yaml(
                make_variation_overrides(name),
                output,
                force=options.force,
                description="CM2026 override file",
            )
            print(f"Wrote {output}")
        return

    if options.variation:
        output = options.output or VARIATIONS_DIR / f"{options.variation}.yaml"
        write_yaml(
            make_variation_overrides(options.variation),
            output,
            force=options.force,
            description="CM2026 override file",
        )
        print(f"Wrote {output}")
        return

    defaults_output = options.output or DEFAULTS_OUTPUT
    write_yaml(
        to_builtin(make_cm2026_defaults()),
        defaults_output,
        force=options.force,
        comments=True,
    )
    print(f"Wrote {defaults_output}")
    write_yaml(
        to_builtin(make_cm2026_synth_defaults()),
        SYNTH_DEFAULTS_OUTPUT,
        force=options.force,
        comments=True,
    )
    print(f"Wrote {SYNTH_DEFAULTS_OUTPUT}")
    write_yaml(
        {},
        BASELINE_OUTPUT,
        force=options.force,
        description="CM2026 override file with no changes",
    )
    print(f"Wrote {BASELINE_OUTPUT}")


if __name__ == "__main__":
    main()
