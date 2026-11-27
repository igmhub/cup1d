from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from cup1d.utils.various_dicts import blob_strings_orig, conv_strings


def set_blinding(apply_blinding, seed):
    """Generate deterministic compressed-cosmology blinding offsets.

    Parameters
    ----------
    apply_blinding : bool
        Draw Gaussian offsets when true; return exact zeros otherwise.
    seed : int or SeedSequence
        Seed supplied to NumPy's default random generator.

    Returns
    -------
    dict of str to float
        Offsets for ``Delta2_star``, ``n_star``, and ``alpha_star``.
    """
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

    Parameters
    ----------
    blind : dict
        Additive offsets keyed by compressed cosmology parameter.
    values : mapping or ndarray
        Mutable mapping, structured array, or unnamed blob array to update.
    sign : {1, -1}
        Apply or remove the offsets.

    Returns
    -------
    mapping or ndarray
        The same object after in-place modification.

    Raises
    ------
    TypeError
        If offset or result containers have unsupported types.
    ValueError
        If an unnamed blob array lacks enough columns.
    KeyError
        If an unknown blinding key is supplied.

    Notes
    -----
    Missing cosmology fields are legitimate (for example a result may contain
    only ``Delta2_star`` and ``n_star``), but malformed inputs are not silently
    ignored.  The operation is in-place, matching the historical API.
    """
    if not isinstance(blind, Mapping):
        raise TypeError("blind must be a dictionary of cosmology offsets")

    if isinstance(values, np.ndarray) and values.dtype.names is None:
        # Unnamed blobs hold the leading fields of the legacy order:
        # (Delta2_star, n_star[, alpha_star[, f_star, g_star, H0]]).
        n_columns = values.shape[-1]
        if n_columns < 2 or n_columns > len(blob_strings_orig):
            raise ValueError(
                f"unnamed cosmology array must have 2 to {len(blob_strings_orig)} "
                f"blob columns; got {values.shape}"
            )
        for key, offset in blind.items():
            if key not in blob_strings_orig:
                raise KeyError(f"Unknown blinding parameter: {key}")
            index = blob_strings_orig.index(key)
            if index < n_columns:
                values[..., index] += sign * offset
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
    """Apply cosmology blinding offsets in place.

    Parameters
    ----------
    blind : dict
        Offsets returned by :func:`set_blinding`.
    dict_cosmo : mapping or ndarray
        Mutable cosmology representation to blind.

    Returns
    -------
    mapping or ndarray
        The modified input object.
    """
    return _apply_blinding_offset(blind, dict_cosmo, sign=1)


def apply_unblinding(blind, dict_cosmo):
    """Remove cosmology blinding offsets in place.

    Parameters
    ----------
    blind : dict
        Offsets returned by :func:`set_blinding`.
    dict_cosmo : mapping or ndarray
        Mutable cosmology representation to unblind.

    Returns
    -------
    mapping or ndarray
        The modified input object.
    """
    return _apply_blinding_offset(blind, dict_cosmo, sign=-1)


class Blinding(Mapping):
    """Blinding offsets of the compressed cosmology for a set of data.

    The single place where the offsets are decided (cup1d's likelihood and
    lya_interface both build it): seeded by the ``blinding`` string of the
    first data set that requests blinding, zero offsets otherwise. It is a
    read-only mapping ``{"Delta2_star": ..., "n_star": ..., "alpha_star": ...}``,
    so it can be passed wherever an offsets dict is expected.

    Parameters
    ----------
    offsets : mapping
        Additive offsets keyed by compressed-cosmology name.
    """

    def __init__(self, offsets):
        self._offsets = MappingProxyType({key: float(value) for key, value in offsets.items()})

    @classmethod
    def from_data(cls, data_sets):
        """Offsets for P1D data sets (iterable of objects with ``apply_blinding``
        and ``blinding``), as native cup1d analyses use them."""
        for data in data_sets:
            if data.apply_blinding:
                seed = int.from_bytes(data.blinding.encode("utf-8"), byteorder="big")
                return cls(set_blinding(True, seed))
        return cls(set_blinding(False, 0))

    def __getitem__(self, key):
        return self._offsets[key]

    def __iter__(self):
        return iter(self._offsets)

    def __len__(self):
        return len(self._offsets)

    def __repr__(self):
        return f"Blinding({dict(self._offsets)})"

    @property
    def is_blinded(self):
        """Whether any offset is non-zero."""
        return any(value != 0 for value in self._offsets.values())

    def blind(self, values):
        """Copy of ``values`` (mapping or structured/blob array) with the offsets added."""
        return apply_blinding(self, _copy(values))

    def unblind(self, values):
        """Copy of ``values`` with the offsets removed."""
        return apply_unblinding(self, _copy(values))


def _copy(values):
    """Copy of a mapping (values as float arrays/scalars) or an array."""
    if isinstance(values, np.ndarray):
        return values.copy()
    return {key: np.array(value, dtype=float) if np.ndim(value) else float(value)
            for key, value in values.items()}
