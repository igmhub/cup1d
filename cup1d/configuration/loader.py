"""Utilities for reading cup1d configuration files."""

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def read_config(filename: str | Path) -> dict[str, Any]:
    """Read a YAML configuration file and return its contents."""

    path = Path(filename).expanduser()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration in {path} must be a YAML mapping")

    return config


def restore_runtime_types(config: dict[str, Any]) -> dict[str, Any]:
    """Restore array types used by the analysis from plain YAML lists."""

    array_paths: set[tuple[str, str]] = set()

    parameter_sections = {
        "fid_igm": config.get("igm_params", []),
        "fid_cont": [
            name[2:]
            for name in config.get("fid_cont", {})
            if name.startswith("n_")
        ],
        "fid_syst": [
            name[2:]
            for name in config.get("fid_syst", {})
            if name.startswith("n_")
        ],
    }
    for section, names in parameter_sections.items():
        values = config.get(section, {})
        for name in names:
            if values.get(f"n_{name}", 0) > 0:
                array_paths.add((section, name))
                if f"{name}_znodes" in values:
                    array_paths.add((section, f"{name}_znodes"))

    for name in config.get("cov_factor", {}):
        array_paths.add(("cov_factor", name))

    for section, name in array_paths:
        value = config.get(section, {}).get(name)
        if isinstance(value, list):
            config[section][name] = np.asarray(value)

    return config


def print_config_values(config: dict[str, Any], source: str) -> None:
    """Print the source and value of every configuration leaf."""

    for name, value in config.items():
        _print_values(name, value, source=source)


def print_resolved_values(
    config: dict[str, Any], overrides: dict[str, Any], prefix: str = ""
) -> None:
    """Print resolved leaves as default or explicitly user-provided."""

    for name, value in config.items():
        full_name = f"{prefix}{name}"
        user_value = overrides.get(name, _MISSING)
        if isinstance(value, dict):
            child_overrides = user_value if isinstance(user_value, dict) else {}
            print_resolved_values(value, child_overrides, prefix=f"{full_name}.")
        else:
            source = "user-provided" if user_value is not _MISSING else "default"
            print(f"Using {source} for {full_name}: {value}")


_MISSING = object()


def apply_overrides(
    defaults: dict[str, Any],
    overrides: dict[str, Any],
    prefix: str = "",
    verbose: bool = True,
) -> dict[str, Any]:
    """Recursively apply user overrides to resolved default values.

    A message is printed for every leaf value, identifying whether it came
    from the CM2026 defaults or from the user configuration.
    """

    delete_keys = overrides.get("__delete__", [])
    if not isinstance(delete_keys, list) or not all(
        isinstance(key, str) for key in delete_keys
    ):
        raise TypeError(f"Option {prefix}__delete__ must be a list of names")

    missing_delete_keys = set(delete_keys) - defaults.keys()
    if missing_delete_keys:
        names = ", ".join(sorted(missing_delete_keys))
        raise ValueError(f"Cannot delete unknown option(s) in {prefix}: {names}")

    additional = overrides.keys() - defaults.keys() - {"__delete__"}
    if additional and not prefix:
        location = "configuration"
        names = ", ".join(sorted(additional))
        raise ValueError(f"Unknown option(s) in {location}: {names}")

    resolved: dict[str, Any] = {}
    for key, default_value in defaults.items():
        name = f"{prefix}{key}"

        if key in delete_keys:
            if verbose:
                print(f"Using user-provided deletion for {name}")
            continue

        if key not in overrides:
            resolved[key] = default_value
            if verbose:
                _print_values(name, default_value, source="default")
            continue

        user_value = overrides[key]
        if isinstance(default_value, dict):
            if not isinstance(user_value, dict):
                raise TypeError(f"Option {name} must be a YAML mapping")
            resolved[key] = apply_overrides(
                default_value,
                user_value,
                prefix=f"{name}.",
                verbose=verbose,
            )
        else:
            resolved[key] = _coerce_like(user_value, default_value)
            if verbose:
                _print_values(name, resolved[key], source="user-provided")

    # Model dictionaries are extensible: some supported variations introduce
    # values that do not exist in the baseline (for example HCD_const_znodes).
    for key in additional:
        name = f"{prefix}{key}"
        resolved[key] = _coerce_additional(overrides[key], name)
        if verbose:
            _print_values(name, resolved[key], source="user-provided")

    return resolved


def _coerce_like(value: Any, default: Any) -> Any:
    """Preserve selected runtime types when replacing YAML values."""

    if isinstance(default, np.ndarray):
        return np.asarray(value, dtype=default.dtype)
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _coerce_additional(value: Any, name: str) -> Any:
    """Coerce extensible model values that have no baseline counterpart."""

    if name.endswith("_znodes"):
        return np.asarray(value)
    return value


def _print_values(name: str, value: Any, source: str) -> None:
    """Print one status line per resolved leaf value."""

    if isinstance(value, dict):
        for child_name, child_value in value.items():
            _print_values(f"{name}.{child_name}", child_value, source)
        return

    print(f"Using {source} for {name}: {value}")
