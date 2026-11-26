"""Smooth version of a measured P1D, cached next to the measurement.

The emulator error enters the likelihood as a relative covariance times
``P_smooth(k_i) P_smooth(k_j)``. Scaling by the noisy measurement would make
the covariance depend on the data noise, so every data reader uses a smooth
curve computed here: per redshift bin, a penalized cubic spline in
``(ln k, ln P)`` with diagonal inverse-variance weights (from the smooth curve
itself, iterated, so that noise does not bias it) and smoothing
``LAM_FACTOR`` times the generalized cross-validation (GCV) optimum (a factor
below one follows the points a bit more closely than GCV), with log(lambda)
replaced by its running median over neighbouring redshifts so that a bin where
GCV fails gets the smoothing of its neighbours.

The spline is fitted to the whole measurement in the file, before any scale
cut, so the cuts do not create border effects; readers then select the rows
they keep. Rows with non-finite values, non-positive variance or non-positive
P1D do not enter the fit but still get the spline value.

The result is cached in ``<measurement>.smooth_p1d.npz`` next to the
measurement file, under a hash of the inputs ``(z, k, P, var)`` and of
:data:`METHOD`; the inputs are stored too, so the cache can be inspected and
the smoothing redone (see ``notebooks/data/observations/smooth_p1d_qmle3``).
A different data bias or covariance adds an entry; changing the method changes
:data:`METHOD` and so the key. Only MPI rank 0 writes the cache.
"""

import hashlib
import os
import tempfile

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import minimize_scalar

from cup1d.utils.warn_once import mpi_rank, warn_once

LAM_FACTOR = 0.3
N_WEIGHT_ITERATIONS = 4
METHOD = f"smoothing-spline-lnk-lnP-gcv-x{LAM_FACTOR}-zmedian3-full-measurement-iterw{N_WEIGHT_ITERATIONS}-v3"
CACHE_SUFFIX = ".smooth_p1d.npz"
FIELDS = ("z", "k", "pk", "var", "smooth")


def gcv_log_lambda(x, y, w):
    """log10 of the smoothing that minimizes generalized cross-validation."""
    eye = np.eye(len(x))

    def gcv(log_lam):  # the spline is linear in y: smoother matrix from the unit vectors
        hat = make_smoothing_spline(x, eye, w=w, lam=10**log_lam)(x)
        r = y - hat @ y
        return len(x) * np.sum(w * r**2) / (len(x) - np.trace(hat)) ** 2

    grid = np.linspace(-8, 2, 41)  # coarse grid, then refine
    start = grid[np.argmin([gcv(g) for g in grid])]
    return minimize_scalar(gcv, bounds=(start - 0.25, start + 0.25), method="bounded").x


class _Bin:
    """Rows of one redshift bin that enter the fit, in ln k order."""

    def __init__(self, k, pk, var):
        k, pk, var = (np.asarray(a, dtype=float) for a in (k, pk, var))
        use = np.isfinite(k) & np.isfinite(pk) & np.isfinite(var) & (var > 0) & (pk > 0) & (k > 0)
        order = np.argsort(k[use])
        self.k = k
        self.x, self.y = np.log(k[use])[order], np.log(pk[use])[order]
        self.var = var[use][order]
        self.level = pk[use][order]  # P that sets the weights, updated by the fit
        self.spline = None

    def weights(self):
        """Inverse variance of ln P = P^2 / var, normalized."""
        w = self.level**2 / self.var
        return w / w.mean()

    def fit(self, log_lam):
        self.spline = make_smoothing_spline(self.x, self.y, w=self.weights(), lam=10**log_lam)
        self.level = np.exp(self.spline(self.x))

    def __call__(self):
        return np.exp(self.spline(np.log(np.where(self.k > 0, self.k, np.nan))))


def running_median(values, width=3):
    """Median of every value and its neighbours (fewer at the ends)."""
    half = width // 2
    return np.array([np.median(values[max(0, i - half): i + half + 1]) for i in range(len(values))])


def smooth_rows(z, k, pk, var, lam_factor=LAM_FACTOR, z_median=True,
                n_iterations=N_WEIGHT_ITERATIONS, return_log_lambda=False):
    """Smooth P1D of a measurement stored one row per (z, k).

    Every redshift bin gets its own spline. Its smoothing is ``lam_factor``
    times the GCV optimum, with log10(lambda) replaced by the running median of
    the bin and its two neighbours in z: GCV can fail in a single bin (at
    z = 3.0 in DESI QMLE3 it interpolates the data), and the smoothing should
    change smoothly with z. The weights are recomputed from the smooth curve
    :data:`N_WEIGHT_ITERATIONS` times; with the measured P they favour upward
    fluctuations and bias the curve high where errors are large (high z).

    Rows with non-finite values, ``var <= 0`` or ``pk <= 0`` do not enter the
    fit but still get the spline value. ``z_median=False`` and
    ``n_iterations=1`` switch off the running median and the weight iteration
    (for comparisons; the cached baseline uses the defaults).

    Returns
    -------
    ndarray
        Smooth P1D of every row; with ``return_log_lambda``, also the
        redshifts and the GCV and adopted log10(lambda) of the last iteration.
    """
    z, k, pk, var = (np.asarray(a, dtype=float) for a in (z, k, pk, var))
    zbins = np.unique(z[np.isfinite(z)])
    bins = [_Bin(k[z == zb], pk[z == zb], var[z == zb]) for zb in zbins]
    for _ in range(n_iterations):
        log_gcv = np.array([gcv_log_lambda(b.x, b.y, b.weights()) for b in bins])
        log_lam = (running_median(log_gcv) if z_median else log_gcv) + np.log10(lam_factor)
        for b, ll in zip(bins, log_lam):
            b.fit(ll)
    out = np.full(len(z), np.nan)
    for zb, b in zip(zbins, bins):
        out[z == zb] = b()
    if return_log_lambda:
        return out, zbins, log_gcv + np.log10(lam_factor), log_lam
    return out


def cache_path(source_file):
    """Cache file of a measurement file (same folder)."""
    return os.fspath(source_file) + CACHE_SUFFIX


def _key(z, k, pk, var):
    """Hash of the inputs and of the method."""
    digest = hashlib.sha1(METHOD.encode())
    for array in (z, k, pk, var):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()[:16]


def read_cache(source_file):
    """Entries of the cache of a measurement file: {key: {field: array}}."""
    path = cache_path(source_file)
    if not os.path.isfile(path):
        return {}
    try:
        with np.load(path) as stored:
            flat = {name: stored[name] for name in stored.files}
    except Exception as error:  # corrupted cache: recompute
        warn_once(f"cannot read {path} ({error}); recomputing the smooth P1D", key=f"smooth-read-{path}")
        return {}
    keys = {name.rsplit("_", 1)[0] for name in flat}
    return {key: {f: flat[f"{key}_{f}"] for f in FIELDS} for key in keys
            if all(f"{key}_{f}" in flat for f in FIELDS)}


def smooth_measurement(source_file, z, k, pk, var):
    """Smooth P1D of a whole measurement, read from or added to the cache.

    Parameters
    ----------
    source_file : str or path-like
        Measurement file; the cache is ``<source_file>.smooth_p1d.npz``.
    z, k, pk, var : array_like
        One row per (z, k) of the full measurement, before scale cuts: redshift,
        wavenumber, P1D and its variance.

    Returns
    -------
    ndarray
        Smooth P1D of every row.
    """
    z, k, pk, var = (np.asarray(a, dtype=float).ravel() for a in (z, k, pk, var))
    key = _key(z, k, pk, var)
    cache = read_cache(source_file)
    if key in cache:
        return np.array(cache[key]["smooth"])
    smooth = smooth_rows(z, k, pk, var)
    if mpi_rank() == 0:
        cache[key] = dict(z=z, k=k, pk=pk, var=var, smooth=smooth)
        _write_atomic(cache_path(source_file), cache)
    return smooth


def smooth_per_redshift(source_file, z_list, k_list, pk_list, var_list):
    """:func:`smooth_measurement` for data stored as per-redshift lists."""
    sizes = [len(k) for k in k_list]
    z = np.repeat(np.asarray(z_list, dtype=float), sizes)
    smooth = smooth_measurement(source_file, z, np.concatenate(k_list), np.concatenate(pk_list),
                                np.concatenate(var_list))
    return np.split(smooth, np.cumsum(sizes)[:-1])


def _write_atomic(path, cache):
    """Write the cache through a temporary file, so readers never see half a file."""
    flat = {f"{key}_{f}": entry[f] for key, entry in cache.items() for f in FIELDS}
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".npz")
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **flat)
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; the cache is shared data
        os.replace(tmp, path)
    except OSError as error:
        if tmp is not None and os.path.exists(tmp):
            os.remove(tmp)
        warn_once(f"cannot write the smooth P1D cache {path} ({error}); it will be recomputed next time",
                  key=f"smooth-write-{path}")
