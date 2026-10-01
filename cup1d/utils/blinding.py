import numpy as np
from cup1d.utils.various_dicts import blob_strings_orig, conv_strings


def set_blinding(apply_blinding, seed):
    """Set the blinding parameters"""
    blind_prior = {"Delta2_star": 0.05, "n_star": 0.01, "alpha_star": 0.005}
    if apply_blinding:
        rng = np.random.default_rng(seed)
    blind = {}
    for key in blind_prior:
        if apply_blinding:
            blind[key] = rng.normal(0, blind_prior[key])
        else:
            blind[key] = 0
    return blind


def _apply_blinding_offset(blind, values, sign):
    """Apply signed offsets to all present cosmology representations.

    Missing cosmology fields are legitimate (for example a result may contain
    only ``Delta2_star`` and ``n_star``), but malformed inputs are not silently
    ignored.  The operation is in-place, matching the historical API.
    """
    if not isinstance(blind, dict):
        raise TypeError("blind must be a dictionary of cosmology offsets")

    if isinstance(values, np.ndarray) and values.dtype.names is None:
        if values.shape[-1] < len(blob_strings_orig):
            raise ValueError("unnamed cosmology array has too few blob columns")
        for key, offset in blind.items():
            if key not in blob_strings_orig:
                raise KeyError(f"Unknown blinding parameter: {key}")
            values[..., blob_strings_orig.index(key)] += sign * offset
        return values

    names = values.dtype.names if isinstance(values, np.ndarray) else None
    if names is None and not hasattr(values, "__contains__"):
        raise TypeError("cosmology values must be a mapping or a structured array")
    for key, offset in blind.items():
        if key not in conv_strings:
            raise KeyError(f"Unknown blinding parameter: {key}")
        for representation in (key, conv_strings[key]):
            present = representation in names if names is not None else representation in values
            if present:
                values[representation] += sign * offset
    return values


def apply_blinding(blind, dict_cosmo):
    """Apply cosmology blinding offsets in-place to every present field."""
    return _apply_blinding_offset(blind, dict_cosmo, sign=1)


def apply_unblinding(blind, dict_cosmo):
    """Remove cosmology blinding offsets in-place from every present field."""
    return _apply_blinding_offset(blind, dict_cosmo, sign=-1)
