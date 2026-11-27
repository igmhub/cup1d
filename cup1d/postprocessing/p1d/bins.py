"""Selected data/model P1D bins of a likelihood, ready for the renderers.

The likelihood owns predictions and chi-squared calculations. This module
only selects their outputs (:class:`P1DPlotter`) into :class:`P1DBin` items,
which every P1D renderer (:mod:`cup1d.postprocessing.p1d.plots`) draws.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2 as chi2_distribution


@dataclass
class P1DBin:
    """One selected dataset/redshift bin in native velocity units."""

    key: str
    index: int
    z: float
    k: np.ndarray
    data: np.ndarray
    model: np.ndarray
    error: np.ndarray
    chi2: float
    dof: int
    color: object
    shift: float
    model_error: np.ndarray | None = None
    realizations: np.ndarray | None = None

    @property
    def probability(self):
        """Return the chi-squared survival probability for this P1D bin.

        Returns
        -------
        float
            Upper-tail probability evaluated from :attr:`chi2` and :attr:`dof`.
        """
        return chi2_distribution.sf(self.chi2, self.dof)


class P1DPlotter:
    """Prepare likelihood predictions independently of figure layout.

    ``values`` is a sampling point, or one point per selected redshift when
    ``z_at_time=True``. Redshifts are ordered as in the input datasets, with
    duplicates removed. ``rand_posterior`` is a two-dimensional array of
    sampling points used to estimate pointwise model uncertainty.
    """

    def __init__(self, likelihood):
        """Bind a P1D plotting adapter to one likelihood.

        Parameters
        ----------
        likelihood : object
            Re-binned likelihood exposing P1D data, predictions, covariances,
            and chi-squared evaluation.
        """
        self.like = likelihood

    def _predict(self, values):
        """Evaluate a P1D prediction and reject out-of-domain points.

        Parameters
        ----------
        values : array_like or mapping
            Sampling point accepted by the likelihood.

        Returns
        -------
        mapping
            Predicted P1D vectors grouped by dataset and redshift index.

        Raises
        ------
        ValueError
            If the sampling point is outside the admitted model domain.
        """
        result = self.like.get_p1d_kms(values)
        if result is None:
            raise ValueError("Cannot plot P1D: the sampling point is outside the model domain.")
        return result[0]

    def prepare(self, values=None, *, zmask=None, plot_every_iz=1,
                z_at_time=False, rand_posterior=None, n_perturb=0,
                collapse=False, plot_panels=False, glob_full=False,
                fix_cosmo=False, n_param_glob_full=16, chi2_nozcov=False):
        """Prepare selected P1D bins, predictions, diagnostics, and draws.

        Parameters
        ----------
        values : array_like, mapping, or sequence, optional
            Joint sampling point, or one point per selected redshift when
            ``z_at_time`` is true.
        zmask : array_like, optional
            Redshifts selected with a tolerance of ``1e-3``.
        plot_every_iz : int, default: 1
            Retain every nth selected redshift bin.
        z_at_time : bool, default: False
            Use a separately fitted point for every selected redshift.
        rand_posterior : ndarray or sequence of mapping, optional
            Posterior points used for pointwise model-error bands.  Numeric
            arrays have shape ``(n_samples, n_free_parameters)``.
        n_perturb : int, default: 0
            Number of joint covariance realizations for each selected dataset.
        collapse, plot_panels : bool
            Control redshift offsets and panel-color convention.
        glob_full : bool, default: False
            Use ``n_param_glob_full`` in per-bin degrees of freedom.
        fix_cosmo : bool, default: False
            Subtract two fixed cosmological parameters from joint degrees of
            freedom.
        n_param_glob_full : int, default: 16
            Parameter count used with ``glob_full``.
        chi2_nozcov : bool, default: False
            Sum independent per-redshift chi-squared values for the displayed
            total.

        Returns
        -------
        tuple
            Selected :class:`P1DBin` objects, total chi-squared, and degrees of
            freedom.

        Raises
        ------
        ValueError
            If sampling input shape, selection, perturbation count, or model
            prediction shape is invalid.
        """
        if not isinstance(plot_every_iz, (int, np.integer)) or plot_every_iz < 1:
            raise ValueError("plot_every_iz must be a positive integer.")
        if not isinstance(n_perturb, (int, np.integer)) or n_perturb < 0:
            raise ValueError("n_perturb must be a nonnegative integer.")
        mask = None if zmask is None else np.atleast_1d(zmask)
        selected = {
            key: [i for i, z in enumerate(data.z)
                  if mask is None or np.any(np.isclose(z, mask, atol=1e-3, rtol=0))]
            for key, data in self.like.data.items()
        }
        if not any(selected.values()):
            raise ValueError("No data redshifts match zmask.")
        points = {}
        if z_at_time:
            redshifts = list(dict.fromkeys(
                float(self.like.data[key].z[i])
                for key, indices in selected.items() for i in indices
            ))
            if (
                isinstance(values, (list, tuple))
                and len(values) == len(redshifts)
                and all(isinstance(point, Mapping) for point in values)
            ):
                points = dict(zip(redshifts, values))
            else:
                array = np.asarray(values)
                if array.shape != (len(redshifts), len(self.like.free_params)):
                    raise ValueError(
                        "z_at_time requires one sampling point per selected redshift."
                    )
                points = dict(zip(redshifts, array))
            predictions = {z: self._predict(point) for z, point in points.items()}
            per_z = {
                z: self.like.get_chi2(
                    point, return_all=True, zmask=np.array([z])
                )[1]
                for z, point in points.items()
            }
            chi2_bins = {
                key: {
                    i: per_z[float(self.like.data[key].z[i])][key][i]
                    for i in indices
                }
                for key, indices in selected.items()
            }
            total_chi2 = sum(sum(v.values()) for v in chi2_bins.values())
        else:
            prediction = self._predict(values)
            # Inference accepts only one zmask value because a multi-redshift
            # subset discards retained cross-redshift covariance. Plot
            # selection remains useful, so collect its diagnostic chi2 values
            # one redshift at a time.
            if mask is not None and len(mask) > 1:
                chi2_bins = {key: {} for key in self.like.data}
                total_chi2 = 0.0
                for z in dict.fromkeys(float(z) for z in mask):
                    chi2_z, each_z = self.like.get_chi2(
                        values, return_all=True, zmask=np.array([z])
                    )
                    total_chi2 += chi2_z
                    for key, indices in selected.items():
                        for i in indices:
                            if np.isclose(
                                self.like.data[key].z[i], z, atol=1e-3, rtol=0
                            ):
                                chi2_bins[key][i] = each_z[key][i]
            else:
                total_chi2, chi2_bins = self.like.get_chi2(
                    values, return_all=True, zmask=mask
                )
            predictions = None
            if chi2_nozcov:
                total_chi2 = sum(chi2_bins[key][i] for key, indices in selected.items() for i in indices)

        posterior = None
        if rand_posterior is not None:
            if (isinstance(rand_posterior, (list, tuple)) and rand_posterior
                    and all(isinstance(sample, Mapping) for sample in rand_posterior)):
                samples = rand_posterior
            else:
                samples = np.asarray(rand_posterior)
                if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] != len(self.like.free_params):
                    raise ValueError("rand_posterior must have shape (n_samples, n_free_parameters).")
            posterior = [self._predict(sample) for sample in samples]

        bins = []
        total_ndata = 0
        self.dataset_stats = {}
        per_redshift_dof = 0
        n_free = len(self.like.free_params)
        for key, indices in selected.items():
            data = self.like.data[key]
            models = {i: np.asarray((predictions[float(data.z[i])] if z_at_time else prediction)[key][i]).reshape(-1)
                      for i in indices}
            noise = self._realizations(key, models, indices, n_perturb)
            for i in indices:
                ndata = np.count_nonzero(data.Pk_kms[i])
                total_ndata += ndata
                stats = self.dataset_stats.setdefault(key, [0.0, 0])
                stats[0] += float(chi2_bins[key][i])
                stats[1] += ndata
                if z_at_time:
                    point = points[float(data.z[i])]
                    if isinstance(point, Mapping):
                        nparameters = np.count_nonzero(list(point.values()))
                    else:
                        nparameters = np.count_nonzero(point)
                else:
                    nparameters = n_free
                dof = ndata - nparameters
                per_redshift_dof += dof
                if glob_full:
                    dof = ndata - n_param_glob_full
                if i % plot_every_iz:
                    continue
                k = np.asarray(data.k_kms[i])
                model = models[i]
                if model.shape != k.shape:
                    raise ValueError(f"P1D prediction shape mismatch for dataset {key}, z={data.z[i]}.")
                model_error = None
                if posterior is not None:
                    model_error = np.std([np.asarray(p[key][i]).reshape(-1) for p in posterior], axis=0)
                nz = len(data.z)
                color = plt.cm.jet(i / (nz - 1)) if nz > 1 and not plot_panels else 'C0'
                shift = 4 * i / (nz - 1) if nz > 1 and not collapse and not plot_panels else 0
                bins.append(P1DBin(
                    key, i, float(data.z[i]), k, np.asarray(data.Pk_kms[i]), model,
                    np.sqrt(np.diag(self.like.cov_Pk_kms[key][i])),
                    float(chi2_bins[key][i]), dof, color, shift, model_error, noise.get(i),
                ))
        if not bins:
            raise ValueError("No redshift bins remain after applying plot_every_iz and zmask.")
        dof = per_redshift_dof if z_at_time else total_ndata - n_free - (2 if fix_cosmo else 0)
        self.ndata = total_ndata
        return bins, float(total_chi2), dof

    def _realizations(self, key, models, indices, count):
        """Draw correlated P1D realizations across selected redshift blocks.

        Parameters
        ----------
        key : str
            Dataset key.
        models : mapping[int, ndarray]
            P1D model vectors in ``km / s`` indexed by redshift bin.
        indices : sequence of int
            Selected redshift-bin indices.
        count : int
            Number of multivariate Gaussian draws.

        Returns
        -------
        dict[int, ndarray]
            Each selected index maps to draws with shape ``(count, nk)``;
            returns an empty mapping when no draws are requested.
        """
        if not count or not indices:
            return {}
        data = self.like.data[key]
        lengths = [len(k) for k in data.k_kms]
        offsets = np.cumsum([0] + lengths)
        chosen = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in indices])
        covariance = self.like.full_cov_Pk_kms[key][np.ix_(chosen, chosen)]
        draws = np.random.multivariate_normal(np.concatenate([models[i] for i in indices]), covariance, count)
        split = np.cumsum([len(models[i]) for i in indices])[:-1]
        return dict(zip(indices, np.split(draws, split, axis=1)))
