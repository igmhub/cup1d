"""Weighted posterior samples in native cup1d parameter names."""

from dataclasses import dataclass, field
import os
import re

import numpy as np

from cup1d.postprocessing.posterior.stats import weighted_correlation, weighted_quantile

from cup1d.postprocessing.labels import STAR_PARAMETERS, parameter_label  # noqa: F401


def family_of(name):
    """Return the history family of a native coefficient (``tau_eff_2`` -> ``tau_eff``).

    Names without a trailing integer index are their own family.
    """
    match = re.fullmatch(r"(.+)_(\d+)", name)
    return match.group(1) if match else name


def index_of(name):
    """Return the trailing integer index of a native coefficient, or None."""
    match = re.fullmatch(r".+_(\d+)", name)
    return int(match.group(1)) if match else None


@dataclass
class PosteriorSamples:
    """Weighted posterior samples of physical parameters.

    Parameters
    ----------
    values : numpy.ndarray, shape (n_samples, n_parameters)
        Physical parameter values (not unit-cube coordinates).
    names : list of str
        Native cup1d name of every column (``As``, ``tau_eff_0``,
        ``HCD_damp1_0``, ``R_coeff_3``, ...). The compressed parameters are
        ``Delta2_star`` and ``n_star``.
    weights : numpy.ndarray, shape (n_samples,), optional
        Sample weights (Cobaya multiplicities); unit weights by default.
    bounds : dict, optional
        ``{name: (min, max)}`` flat-prior bounds of the sampled parameters.
    minuslogpost : numpy.ndarray, shape (n_samples,), optional
        Minus log posterior of every sample.
    sampled : list of str, optional
        Names of the sampled parameters (by default every name with bounds).
    label : str, default=""
        Legend label of this posterior.
    blinded : bool, default=False
        Whether the star parameters include blinding offsets.

    Raises
    ------
    ValueError
        If shapes disagree or a name is repeated.
    """

    values: np.ndarray
    names: list
    weights: np.ndarray | None = None
    bounds: dict = field(default_factory=dict)
    minuslogpost: np.ndarray | None = None
    sampled: list | None = None
    label: str = ""
    blinded: bool = False

    def __post_init__(self):
        self.values = np.asarray(self.values)
        self.names = list(self.names)
        if self.values.ndim != 2 or self.values.shape[1] != len(self.names):
            raise ValueError(f"values must have shape (n_samples, {len(self.names)})")
        if len(set(self.names)) != len(self.names):
            raise ValueError("repeated parameter names")
        n = self.values.shape[0]
        self.weights = np.ones(n) if self.weights is None else np.asarray(self.weights, dtype=float)
        if self.weights.shape != (n,) or np.any(self.weights < 0) or not self.weights.sum() > 0:
            raise ValueError("weights must be non-negative with shape (n_samples,) and positive sum")
        if self.minuslogpost is not None:
            self.minuslogpost = np.asarray(self.minuslogpost, dtype=float)
            if self.minuslogpost.shape != (n,):
                raise ValueError("minuslogpost must have shape (n_samples,)")
        if self.sampled is None:
            self.sampled = [name for name in self.names if name in self.bounds]
        self._index = {name: i for i, name in enumerate(self.names)}

    def __contains__(self, name):
        return name in self._index

    def __len__(self):
        return self.values.shape[0]

    @property
    def effective_size(self):
        """Kish effective sample size ``(sum w)^2 / sum w^2``."""
        return self.weights.sum() ** 2 / np.sum(self.weights ** 2)

    def column(self, name):
        """Return the samples of one parameter as float64.

        Raises
        ------
        KeyError
            If the parameter is not in the samples.
        """
        return self.values[:, self._index[name]].astype(float)

    def columns(self, names=None):
        """Return ``{name: samples}`` for the given (default: all) parameters."""
        return {name: self.column(name) for name in (self.names if names is None else names)}

    def family(self, family):
        """Return the names of a history family ordered by index (``tau_eff`` -> ``tau_eff_0``...)."""
        members = [name for name in self.names if family_of(name) == family and index_of(name) is not None]
        return sorted(members, key=index_of)

    def quantiles(self, name, quantiles=(0.16, 0.5, 0.84)):
        """Return weighted quantiles of one parameter."""
        return weighted_quantile(self.column(name), quantiles, self.weights)

    def correlation(self, names):
        """Return the weighted correlation matrix of the given parameters."""
        return weighted_correlation(np.column_stack([self.column(n) for n in names]), self.weights)

    def label_of(self, name):
        """LaTeX label of a parameter, marking blinded star parameters."""
        return parameter_label(name, self.blinded)

    def subsample(self, n_samples, seed=0):
        """Draw unweighted samples with probability proportional to the weights.

        Parameters
        ----------
        n_samples : int
            Number of draws (with replacement); all rows are returned
            unchanged when there are fewer rows than draws and the weights
            are uniform.
        seed : int, default=0
            Seed of the random generator.

        Returns
        -------
        PosteriorSamples
            Unit-weight samples with the same names and metadata.
        """
        if len(self) <= n_samples and np.all(self.weights == self.weights[0]):
            return self
        rng = np.random.default_rng(seed)
        rows = rng.choice(len(self), size=n_samples, replace=True, p=self.weights / self.weights.sum())
        return PosteriorSamples(
            self.values[rows], self.names, None, self.bounds,
            None if self.minuslogpost is None else self.minuslogpost[rows],
            self.sampled, self.label, self.blinded,
        )

    def best_sample(self):
        """Return ``{name: value}`` of the sample with the lowest minus log posterior.

        Raises
        ------
        ValueError
            If ``minuslogpost`` is not available.
        """
        if self.minuslogpost is None:
            raise ValueError("no minuslogpost stored with these samples")
        row = int(np.argmin(self.minuslogpost))
        return {name: float(self.values[row, i]) for i, name in enumerate(self.names)}

    def near_bounds(self, fraction=0.02, threshold=0.05):
        """List sampled parameters with posterior mass close to a prior bound.

        Parameters
        ----------
        fraction : float, default=0.02
            Width of the edge region as a fraction of the prior range.
        threshold : float, default=0.05
            Minimum posterior mass in an edge region to be listed.

        Returns
        -------
        list of tuple
            ``(name, mass_lower, mass_upper)``, largest mass first.
        """
        rows = []
        total = self.weights.sum()
        for name in self.sampled:
            low, high = self.bounds[name]
            width = fraction * (high - low)
            x = self.column(name)
            lower = self.weights[x < low + width].sum() / total
            upper = self.weights[x > high - width].sum() / total
            if max(lower, upper) > threshold:
                rows.append((name, lower, upper))
        return sorted(rows, key=lambda row: -max(row[1:]))

    @classmethod
    def from_cup1d_folder(cls, folder, nburn_extra=0, blinding=None, blinded=False, label="", thin=1):
        """Load the samples of a native cup1d (emcee) run.

        Parameters
        ----------
        folder : str or path-like
            Folder with ``chain.npy`` (unit-cube walkers), ``blobs.npy``,
            ``lnprob.npy`` and ``sampler_results.npy`` (or the legacy
            ``fitter_results.npy``).
        nburn_extra : int, default=0
            Extra initial steps to discard.
        blinding : mapping, optional
            ``{"Delta2_star": offset, "n_star": offset}`` subtracted from the
            stored star parameters (e.g. ``data/blinding_dr1.npy``). Without
            it the star parameters stay as stored.
        blinded : bool, default=False
            Mark the stored star parameters as blinded (ignored when
            ``blinding`` is given, which unblinds them).
        label : str, default=""
            Legend label.
        thin : int, default=1
            Keep every ``thin``-th step of every walker. The chain is memory
            mapped, so only the kept steps are read.

        Returns
        -------
        PosteriorSamples
        """
        from cup1d.postprocessing.chains.products import load_parameter_bounds
        from cup1d.utils.various_dicts import get_blob_value

        names, bounds = load_parameter_bounds(folder)
        steps = slice(nburn_extra, None, thin)
        chain = np.load(os.path.join(folder, "chain.npy"), mmap_mode="r")[steps]
        blobs = np.load(os.path.join(folder, "blobs.npy"), mmap_mode="r")[steps]
        lnprob = np.load(os.path.join(folder, "lnprob.npy"))[steps]
        n_rows = chain.shape[0] * chain.shape[1]
        values = np.empty((n_rows, len(names) + len(STAR_PARAMETERS)))
        offsets = blinding or {}
        for i, name in enumerate(STAR_PARAMETERS):
            values[:, i] = np.asarray(get_blob_value(blobs, name)).reshape(-1) - offsets.get(name, 0.0)
        low = np.array([bounds[n][0] for n in names])
        high = np.array([bounds[n][1] for n in names])
        values[:, len(STAR_PARAMETERS):] = np.asarray(chain).reshape(n_rows, -1)
        values[:, len(STAR_PARAMETERS):] *= high - low
        values[:, len(STAR_PARAMETERS):] += low
        return cls(
            values, [*STAR_PARAMETERS, *names],
            bounds=bounds, minuslogpost=-lnprob.reshape(-1), sampled=names,
            label=label, blinded=bool(blinded) and blinding is None,
        )
