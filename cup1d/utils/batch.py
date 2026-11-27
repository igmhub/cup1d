"""Parameter points as batches: a single point is a batch of one.

Every model in cup1d evaluates columns of parameters, ``{name: (n_batch,)}``.
These helpers promote a scalar point (``{name: float}``) to a batch of one
and drop the batch axis from the results again, so there is one
implementation for one point and for many.
"""

import numpy as np


def as_columns(like_params):
    """Parameter columns, batch size, and whether the input was one point.

    Parameters
    ----------
    like_params : mapping or None
        Scalar values (one point) or one-dimensional arrays of equal length
        (a batch). ``None`` or an empty mapping is one point at the fiducial.

    Returns
    -------
    columns : dict of str to ndarray
        Arrays of shape ``(n_batch,)``.
    n_batch : int
    scalar : bool
        True for a single point (the caller drops the batch axis).

    Raises
    ------
    ValueError
        For mixed scalars and arrays, multi-dimensional values or columns of
        different lengths.
    """
    if not like_params:
        return {}, 1, True
    columns = {}
    ndims = set()
    for name, value in like_params.items():
        value = np.asarray(value, dtype=float)
        ndims.add(value.ndim)
        columns[name] = value
    if ndims == {0}:
        return {name: value.reshape(1) for name, value in columns.items()}, 1, True
    if ndims != {1}:
        raise ValueError(
            "likelihood parameters must be all scalars or all one-dimensional batch arrays"
        )
    lengths = {len(value) for value in columns.values()}
    if len(lengths) != 1:
        raise ValueError("all batched likelihood parameters must share n_batch")
    return columns, lengths.pop(), False


def is_batch(like_params):
    """True when ``like_params`` holds one-dimensional columns (see :func:`as_columns`)."""
    return not as_columns(like_params)[2]


def batch_size(columns):
    """Length of the columns (1 for an empty mapping)."""
    for value in columns.values():
        return len(value)
    return 1


def row(columns, index):
    """One point of a batch, as scalar values."""
    return {name: value[index] for name, value in columns.items()}


def unbatch(rows, scalar):
    """Drop the batch axis of per-redshift ``(n_batch, ...)`` rows for one point."""
    return [item[0] for item in rows] if scalar else rows


def as_redshift_rows(z, k_kms):
    """Redshifts as an array and wavenumbers as one array per redshift.

    A single redshift may come with a single wavenumber array.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if len(z) == 1 and np.ndim(k_kms) == 1 and np.ndim(k_kms[0]) == 0:
        k_kms = [k_kms]
    return z, [np.asarray(values, dtype=float) for values in k_kms]
