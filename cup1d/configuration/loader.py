"""Utilities for reading cup1d configuration files."""

import difflib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def read_config(filename: str | Path) -> dict[str, Any]:
    """Read a YAML configuration mapping.

    Parameters
    ----------
    filename : str or pathlib.Path
        YAML file path. User-home syntax is expanded before opening.

    Returns
    -------
    dict
        Parsed YAML root mapping without default resolution or type coercion.

    Raises
    ------
    ValueError
        If the YAML root is not a mapping.
    OSError
        If the file cannot be opened.
    yaml.YAMLError
        If the file is not valid YAML.
    """

    path = Path(filename).expanduser()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration in {path} must be a YAML mapping")

    return config


def restore_runtime_types(config: dict[str, Any]) -> dict[str, Any]:
    """Restore NumPy arrays that YAML represents as lists.

    Parameters
    ----------
    config : dict
        Resolved configuration mutated in place.

    Returns
    -------
    dict
        The same ``config`` mapping, with configured model nodes, fiducial
        histories, and covariance-factor arrays converted from lists.
    """

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
    """Print every leaf of a configuration mapping with one source label.

    Parameters
    ----------
    config : dict
        Possibly nested configuration mapping.
    source : str
        Label printed for every leaf, such as ``"default"`` or
        ``"user-provided"``.
    """

    for name, value in config.items():
        _print_values(name, value, source=source)


def print_resolved_values(
    config: dict[str, Any], overrides: dict[str, Any], prefix: str = ""
) -> None:
    """Print each resolved leaf and whether an override supplied it.

    Parameters
    ----------
    config : dict
        Fully resolved, possibly nested configuration mapping.
    overrides : dict
        User-provided nested values used to identify each leaf's source.
    prefix : str, default=""
        Dotted path already traversed during recursive calls.
    """

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
    warn_unknown: bool = True,
) -> dict[str, Any]:
    """Merge user overrides into a resolved default configuration.

    Parameters
    ----------
    defaults : dict
        Nested default mapping. It is not mutated.
    overrides : dict
        Nested replacement values. ``"__delete__"`` may list direct child
        keys to remove. Additional keys are accepted only in nested model
        dictionaries.
    prefix : str, default=""
        Dotted path used in recursive diagnostics.
    verbose : bool, default=True
        Print each leaf's source or explicit deletion.
    warn_unknown : bool, default=True
        Warn about nested keys absent from the defaults, which are usually
        typos (e.g. ``cov_factor.val_stats``). Known extensions, such as
        ``<parameter>_znodes`` in model sections, ``cov_factor.z`` or entries
        of mappings that are empty by default, do not warn.

    Returns
    -------
    dict
        Newly constructed resolved configuration, preserving selected NumPy
        array and tuple runtime types.

    Raises
    ------
    TypeError
        If a deletion list or nested override has the wrong type.
    ValueError
        If a deletion names an unknown key or a new top-level key is supplied.
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
                warn_unknown=warn_unknown,
            )
        else:
            resolved[key] = _coerce_like(user_value, default_value)
            if verbose:
                _print_values(name, resolved[key], source="user-provided")

    # Model dictionaries are extensible: some supported variations introduce
    # values that do not exist in the baseline (for example HCD_const_znodes).
    for key in additional:
        name = f"{prefix}{key}"
        if warn_unknown and not _is_known_extension(defaults, key, prefix):
            close = difflib.get_close_matches(key, list(defaults), n=1)
            hint = f" Did you mean {prefix}{close[0]}?" if close else ""
            warnings.warn(
                f"Unknown option {name} is not in the defaults and may be a "
                f"typo; it is kept but may be ignored.{hint}",
                UserWarning,
                stacklevel=2,
            )
        resolved[key] = _coerce_additional(overrides[key], name)
        if verbose:
            _print_values(name, resolved[key], source="user-provided")

    return resolved


def _is_known_extension(defaults: dict[str, Any], key: str, prefix: str) -> bool:
    """Return whether a nested key absent from the defaults is a known extension.

    Parameters
    ----------
    defaults : dict
        Default mapping at this nesting level.
    key : str
        Additional user-provided key.
    prefix : str
        Dotted path of this mapping, e.g. ``"cov_factor."``.

    Returns
    -------
    bool
        True for entries of mappings that are empty by default (free-form,
        e.g. ``initial_sampling_values``), ``cov_factor.z``, and, in model
        sections, a parameter's fiducial value (``<name>``) or nodes
        (``<name>_znodes``) when the section configures that parameter
        (e.g. through ``n_<name>`` or ``<name>_fixed``).
    """

    if not defaults:
        return True
    if prefix == "cov_factor." and key == "z":
        return True
    stem = key[: -len("_znodes")] if key.endswith("_znodes") else key
    return any(
        name == f"n_{stem}" or name.startswith(stem + "_") for name in defaults
    )


def _coerce_like(value: Any, default: Any) -> Any:
    """Coerce a replacement to selected types of its default value.

    Parameters
    ----------
    value : Any
        User-provided replacement value.
    default : Any
        Default value whose NumPy-array dtype or tuple type is preserved.

    Returns
    -------
    Any
        Array, tuple, or unchanged replacement according to ``default``.
    """

    if isinstance(default, np.ndarray):
        return np.asarray(value, dtype=default.dtype)
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _coerce_additional(value: Any, name: str) -> Any:
    """Coerce a supported additional nested model value.

    Parameters
    ----------
    value : Any
        User-provided value without a baseline counterpart.
    name : str
        Fully qualified configuration path.

    Returns
    -------
    Any
        NumPy array for ``*_znodes`` values; otherwise the unchanged value.
    """

    if name.endswith("_znodes"):
        return np.asarray(value)
    return value


def _print_values(name: str, value: Any, source: str) -> None:
    """Print recursive configuration leaves with a common source label.

    Parameters
    ----------
    name : str
        Current dotted configuration path.
    value : Any
        Leaf value or nested dictionary to traverse.
    source : str
        Label printed before the value.
    """

    if isinstance(value, dict):
        for child_name, child_value in value.items():
            _print_values(f"{name}.{child_name}", child_value, source)
        return

    print(f"Using {source} for {name}: {value}")
