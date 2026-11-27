import numpy as np


def get_bin_coverage(xmin_o, xmax_o, xmin_n, xmax_n):
    """Compute fractional overlap of old bins with new bins.

    Parameters
    ----------
    xmin_o, xmax_o : array_like
        Lower and upper edges of the old bins.
    xmin_n, xmax_n : array_like
        Lower and upper edges of the target bins.

    Returns
    -------
    ndarray, shape (n_new, n_old)
        Fractions of each old-bin width contributing to each target bin.
    """
    # check out https://stcorp.github.io/harp/doc/html/algorithms/regridding.html
    cover = np.zeros((len(xmin_n), len(xmin_o)))
    for jj in range(len(xmin_n)):
        cover[jj] = np.fmax(
            (np.fmin(xmax_o, xmax_n[jj]) - np.fmax(xmin_o, xmin_n[jj]))
            / (xmax_o - xmin_o),
            0,
        )
    return cover


def _same_redshifts(dict_data):
    """Whether every data set has the same redshift bins, in the same order."""
    zs = [np.atleast_1d(np.asarray(data.z, dtype=float)) for data in dict_data.values()]
    return all(len(z) == len(zs[0]) and np.allclose(z, zs[0], rtol=0, atol=1e-6) for z in zs)


def _separate_step(data, iz, k_rebin_factor):
    """Fine spacing of the separate scheme for one data set and redshift."""
    n = len(data.k_kms_min[iz]) * k_rebin_factor
    return (data.k_kms_max[iz][-1] * 1.05 - data.k_kms_min[iz][0] * 0.95) / (n - 1)


def shared_fine_grid(datasets, iz, k_rebin_factor):
    """Fine k grid shared by several data sets at one redshift.

    The data set with the narrowest bins (the primary) sets the grid as in
    the separate scheme: ``k_rebin_factor`` points per bin, uniform from 0.95
    of its first lower edge to 1.05 of its last upper edge. It is extended
    with the same spacing below, to the lowest bin edge of the other data
    sets, and above, up to the lower edge of the first bin of another data
    set that starts beyond it: every bin that overlaps the primary range is
    then averaged on the primary spacing, and the last extended cell is cut
    at that edge. Beyond it (e.g. high-resolution k above the DESI range)
    the grid has the spacing those data sets have in the separate scheme.

    Parameters
    ----------
    datasets : sequence
        Data objects with ``k_kms_min`` and ``k_kms_max`` per redshift.
    iz : int
        Redshift index, common to all data sets.
    k_rebin_factor : int
        Fine points per data bin.

    Returns
    -------
    k : ndarray
        Strictly increasing fine-cell centres in ``s / km``.
    lower, upper : ndarray
        Fine-cell edges; contiguous, possibly non-uniform across segments.
    """
    widths = [np.median(np.asarray(d.k_kms_max[iz]) - np.asarray(d.k_kms_min[iz])) for d in datasets]
    primary = datasets[int(np.argmin(widths))]
    others = [d for d in datasets if d is not primary]
    k = np.linspace(primary.k_kms_min[iz][0] * 0.95, primary.k_kms_max[iz][-1] * 1.05,
                    len(primary.k_kms_min[iz]) * k_rebin_factor)
    step = k[1] - k[0]
    lower, upper = k - 0.5 * step, k + 0.5 * step
    other_lower = np.concatenate([np.asarray(d.k_kms_min[iz], dtype=float) for d in others])
    other_upper = np.concatenate([np.asarray(d.k_kms_max[iz], dtype=float) for d in others])

    # below: same spacing down to the lowest edge of the other data sets
    n_below = int(np.ceil((lower[0] - other_lower.min()) / step)) if other_lower.min() < lower[0] else 0
    if n_below:
        if lower[0] - n_below * step <= 0:
            raise ValueError("shared fine grid would reach k <= 0")
        k = np.concatenate([k[0] - step * np.arange(n_below, 0, -1), k])
        lower, upper = k - 0.5 * step, k + 0.5 * step

    # above: same spacing up to the lower edge of the next bin of the other
    # data sets (the first one starting beyond the primary grid), last cell
    # cut there. Bins may overlap (e.g. log-spaced high-resolution bins), so a
    # bin straddling that edge is averaged over cells of both segments, which
    # the overlap-length weights handle exactly.
    starts = other_lower[other_lower >= upper[-1]]
    if len(starts):
        boundary = starts.min()
    else:  # no bin starts beyond: cover the remaining ones on this spacing
        boundary = max(upper[-1], other_upper.max())
    if boundary > upper[-1]:
        n_above = int(np.ceil((boundary - upper[-1]) / step))
        new_lower = upper[-1] + step * np.arange(n_above)
        new_upper = np.minimum(new_lower + step, boundary)
        lower = np.concatenate([lower, new_lower])
        upper = np.concatenate([upper, new_upper])

    # beyond: the spacing of the separate scheme of those data sets
    # (k_rebin_factor points per bin over their whole range), up to 1.05 of
    # their last upper edge
    beyond = [d for d in others if d.k_kms_max[iz][-1] > boundary]
    if beyond:
        end = max(d.k_kms_max[iz][-1] for d in beyond) * 1.05
        tail_step = min(_separate_step(d, iz, k_rebin_factor) for d in beyond)
        edges = np.linspace(boundary, end, int(np.ceil((end - boundary) / tail_step)) + 1)
        lower = np.concatenate([lower, edges[:-1]])
        upper = np.concatenate([upper, edges[1:]])
    k = 0.5 * (lower + upper)
    if np.any(np.diff(k) <= 0) or np.any(lower[1:] < upper[:-1] - 1e-15 * upper[:-1]):
        raise ValueError("shared fine grid is not strictly increasing and contiguous")
    return k, lower, upper


class Rebinning(object):
    """Precompute overlap weights for rebinning P1D k grids."""

    def __init__(self, dict_data, k_rebin_factor=1, n_k_emu=None, shared_k_grid=True):
        """Build rebinning grids and overlap matrices for each data set.

        Parameters
        ----------
        dict_data : mapping
            Data objects with ``z``, ``k_kms``, ``k_kms_min``, and
            ``k_kms_max`` attributes.
        k_rebin_factor : int, default: 1
            Number of fine-grid points per original data-grid point. One
            preserves the original grids and identity overlap matrices.
        n_k_emu : int, optional
            When rebinning, also store in ``k_kms_emu`` a grid per redshift of
            ``n_k_emu`` log-spaced points spanning the fine grid. Theory then
            evaluates the emulator on this grid and interpolates its P1D to
            the fine grid, instead of evaluating the emulator at every fine
            point. Log spacing matters: the data bins are linear in k and too
            coarse in ln k at low k for an accurate interpolation.
        shared_k_grid : bool, default: True
            With several data sets at the same redshifts and
            ``k_rebin_factor > 1``, use one fine grid per redshift for all of
            them (:func:`shared_fine_grid`), so the model is evaluated once
            and averaged into every data set's bins. Data sets with different
            redshifts, or ``False``, use one grid per data set.

        Notes
        -----
        ``groups`` lists the data-set keys sharing fine grids (one group with
        every key when shared, otherwise one group per key). On a shared
        grid, fine cells are weighted by their overlap length with each bin,
        since the cells are not all equally wide.
        """

        self.k_kms = {}
        # None when the emulator is evaluated directly on ``k_kms``
        self.k_kms_emu = {}
        self.zs = {}
        self.cover = {}
        self.sum_cover = {}
        keys = list(dict_data)
        self.shared = bool(shared_k_grid) and k_rebin_factor > 1 and len(keys) > 1 and _same_redshifts(dict_data)
        self.groups = [keys] if self.shared else [[key] for key in keys]

        for key in dict_data:
            self.k_kms[key] = []  # new k_kms
            self.cover[key] = []  # to accelerate rebinning
            self.sum_cover[key] = []  # to accelerate rebinning
            self.zs[key] = dict_data[key].z
            self.k_kms_emu[key] = None

        if self.shared:
            for iz in range(len(self.zs[keys[0]])):
                fine, lower, upper = shared_fine_grid(list(dict_data.values()), iz, k_rebin_factor)
                for key in keys:
                    data = dict_data[key]
                    # overlap lengths: the shared cells are not all equally wide
                    _cover = get_bin_coverage(lower, upper, data.k_kms_min[iz], data.k_kms_max[iz]) * (upper - lower)
                    self.k_kms[key].append(fine)
                    self.cover[key].append(_cover)
                    self.sum_cover[key].append(np.sum(_cover, axis=1))
        else:
            for key in keys:
                data = dict_data[key]
                if k_rebin_factor == 1:
                    # Do nothing for a factor of one; retain the original k grid.
                    for iz in range(len(self.zs[key])):
                        self.k_kms[key].append(data.k_kms[iz])
                        self.cover[key].append(
                            np.eye(len(data.k_kms[iz]))
                        )
                        self.sum_cover[key].append(np.sum(self.cover[key][iz], axis=1))
                    continue
                for iz in range(len(self.zs[key])):
                    nelem = len(data.k_kms[iz]) * k_rebin_factor
                    _kms_reb = np.linspace(
                        data.k_kms_min[iz][0] * 0.95,
                        data.k_kms_max[iz][-1] * 1.05,
                        nelem,
                    )
                    self.k_kms[key].append(_kms_reb)
                    xmin_o = _kms_reb - 0.5 * (_kms_reb[1] - _kms_reb[0])
                    xmax_o = _kms_reb + 0.5 * (_kms_reb[1] - _kms_reb[0])

                    _cover = get_bin_coverage(
                        xmin_o,
                        xmax_o,
                        data.k_kms_min[iz],
                        data.k_kms_max[iz],
                    )
                    self.cover[key].append(_cover)
                    self.sum_cover[key].append(np.sum(_cover, axis=1))

        if k_rebin_factor != 1 and n_k_emu is not None:
            for key in keys:
                # identical fine grids give identical (shared) emulator grids
                self.k_kms_emu[key] = [
                    np.geomspace(fine[0], fine[-1], n_k_emu)
                    for fine in self.k_kms[key]
                ]

    def rebinning(self, key, Pk_kms_newk):
        """Average fine-grid P1D predictions onto the data k bins.

        Parameters
        ----------
        key : hashable
            Data-set key supplied to the constructor.
        Pk_kms_newk : sequence of array_like
            One prediction per redshift bin on the fine grid: ``(n_k_fine,)``
            for one point or ``(n_batch, n_k_fine)`` for a batch.

        Returns
        -------
        list of ndarray
            Predictions on the data grids, ``(n_k_data,)`` or
            ``(n_batch, n_k_data)`` per redshift.

        Raises
        ------
        ValueError
            If the redshift count or a fine-grid width does not match.
        """
        if len(Pk_kms_newk) != len(self.zs[key]):
            raise ValueError(
                f"expected {len(self.zs[key])} redshift predictions for {key}; "
                f"got {len(Pk_kms_newk)}"
            )
        weights = self._weights(key)
        rebinned = []
        for iz, prediction in enumerate(Pk_kms_newk):
            prediction = np.asarray(prediction, dtype=float)
            if prediction.ndim not in (1, 2) or prediction.shape[-1] != weights[iz].shape[1]:
                raise ValueError(
                    f"redshift {iz} prediction has shape {prediction.shape}; expected "
                    f"(n_batch,) + ({weights[iz].shape[1]},)"
                )
            rebinned.append(prediction @ weights[iz].T)
        return rebinned

    def _weights(self, key):
        """Rebinning matrices of data set ``key``, one ``(n_k_data, n_k_fine)`` per redshift."""
        if not hasattr(self, "_weight_cache"):
            self._weight_cache = {}
        if key not in self._weight_cache:
            self._weight_cache[key] = [
                cover / total[:, None] for cover, total in zip(self.cover[key], self.sum_cover[key])
            ]
        return self._weight_cache[key]
