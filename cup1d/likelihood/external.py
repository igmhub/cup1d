"""Likelihood of static P1D data for raw P1D predictions from an external theory.

The theory side of the contract (requests, projections) is in
:mod:`cup1d.theory.projection`; the observation model in
:mod:`cup1d.models.observation`.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_solve

from cup1d.utils.rebinning import Rebinning

from cup1d.models.observation import apply_observation_model, apply_resolution, dataset_nuisance, without_resolution
from cup1d.theory.projection import PredictionGroup, PredictionRequest


@dataclass(frozen=True)
class LikelihoodResult:
    """Correlated Gaussian P1D likelihood diagnostics.

    Instances retain per-data-set residuals and chi-squared contributions,
    total data chi-squared, prior contribution, and prediction validity
    information for external likelihood callers.
    """
    loglike: float
    chi2_data: float
    logdet_cov: float
    ndata: int
    valid: bool
    # data-set label -> chi2 and number of points (independent blocks: they
    # add up to chi2_data and ndata)
    chi2_by_dataset: dict = field(default_factory=dict)
    ndata_by_dataset: dict = field(default_factory=dict)


def gaussian_residual(diff, factor, check_finite=True):
    """Evaluate correlated Gaussian residual statistics from a Cholesky factor.

    Parameters
    ----------
    diff : array-like
        One-dimensional data-minus-model residual vector.
    factor : tuple
        Cholesky factor representation accepted by :func:`scipy.linalg.cho_solve`.
    check_finite : bool, default=True
        Check residual finiteness in this function and SciPy calls.

    Returns
    -------
    chi2, logdet : tuple of float
        Quadratic form and covariance log determinant.

    Raises
    ------
    ValueError
        If ``diff`` is not a finite one-dimensional vector.
    """
    diff = np.asarray(diff, dtype=float)
    if diff.ndim != 1 or (check_finite and not np.all(np.isfinite(diff))):
        raise ValueError("residual must be a finite vector")
    return float(diff @ cho_solve(factor, diff, check_finite=check_finite)), float(2 * np.log(np.diag(factor[0])).sum())


class ExternalP1DLikelihood:
    """Evaluate static P1D data against externally supplied raw predictions.

    This boundary owns data rebinning, observational responses, and effective
    covariance.  Theory code supplies only uncontaminated P1D rows satisfying
    the immutable :class:`PredictionRequest` contract.
    """

    def __init__(self, data, model_cont, model_syst, cov_factor, emulator_covariance, fiducial_conversion, emu_cov_type="block", k_rebin_factor=1, configuration_id=None, include_logdet=False, syst_data_labels=None, shared_k_grid=True):
        """Initialize static data response, covariance, and prediction request.

        Parameters
        ----------
        data : mapping
            P1D datasets keyed by identifier.
        model_cont, model_syst : object
            Contaminant and systematic response models.
        cov_factor : mapping
            Redshift-dependent covariance scaling arrays.
        emulator_covariance : mapping
            Emulator covariance metadata with ``zz_zk``, ``k_Mpc_zk``, and
            ``cov_zk`` entries.
        fiducial_conversion : mapping or array-like
            Fiducial velocity-to-comoving conversion information used to
            project emulator covariance.
        emu_cov_type : {'diagonal', 'block', 'full'}, default='block'
            Emulator covariance approximation.
        k_rebin_factor : int, default=1
            Positive rebinning factor applied to data grids.
        configuration_id : str, optional
            Caller-defined configuration identifier embedded in the request.
        include_logdet : bool, default=False
            Include covariance log determinant in likelihood values.
        syst_data_labels : sequence of str, optional
            Data sets that receive the resolution correction, as native
            ``args.syst_data_labels``; ``None`` applies it to all of them.
        shared_k_grid : bool, default=True
            One fine grid per redshift for data sets with the same redshifts,
            as native ``args.shared_k_grid`` (:class:`Rebinning`).

        Raises
        ------
        ValueError
            If data, covariance scaling, covariance metadata, covariance
            matrices, or covariance approximation are invalid.
        """
        from cup1d.likelihood.likelihood import Likelihood
        self.data, self.model_cont, self.model_syst = data, model_cont, model_syst
        if not isinstance(k_rebin_factor, int) or k_rebin_factor < 1 or not data:
            raise ValueError("nonempty data and positive integer rebin factor required")
        self.Rebin_data = Rebinning(data, k_rebin_factor=k_rebin_factor, shared_k_grid=shared_k_grid)
        self.cov_factor = {key: np.asarray(value) for key, value in cov_factor.items()}
        self.emu_cov_type, self.include_logdet = emu_cov_type, include_logdet
        self.syst_data_labels = None if syst_data_labels is None else tuple(syst_data_labels)
        z_factors = self.cov_factor.get("z")
        if z_factors is None or z_factors.ndim != 1 or not len(z_factors) or not np.all(np.isfinite(z_factors)):
            raise ValueError("covariance factors require a finite redshift vector")
        for name in ("val_stat", "val_syst", "val_emu", "val_full"):
            value = self.cov_factor.get(name)
            if value is None or value.shape != z_factors.shape or np.any(~np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"invalid covariance scaling {name}")
        for key, dataset in data.items():
            if dataset.full_Pk_kms is not None:
                expected_z = np.concatenate([np.full(len(k), z) for z, k in zip(dataset.z, dataset.k_kms)])
                if not np.array_equal(dataset.full_zs, expected_z) or not np.array_equal(dataset.full_k_kms, np.concatenate(dataset.k_kms)) or not np.array_equal(dataset.full_Pk_kms, np.concatenate(dataset.Pk_kms)):
                    raise ValueError(f"full data vector ordering differs from redshift blocks for {key}")
        if emu_cov_type not in {"diagonal", "block", "full"}:
            raise ValueError("unknown emulator covariance type")
        for name in ("zz_zk", "k_Mpc_zk", "cov_zk"):
            if name not in emulator_covariance:
                raise ValueError(f"missing emulator covariance metadata {name}")
        matrices = [matrix for dataset in data.values() for matrix in dataset.cov_Pk_kms]
        matrices += [dataset.full_cov_Pk_kms for dataset in data.values() if dataset.full_Pk_kms is not None]
        matrices += [emulator_covariance["cov_zk"]]
        for matrix in matrices:
            matrix = np.asarray(matrix)
            tolerance = 1e-12 * max(np.max(np.abs(matrix)), np.finfo(float).tiny)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, rtol=0, atol=tolerance):
                raise ValueError("covariance must be finite and symmetric")
        Likelihood.set_icov(self, emulator_covariance, fiducial_conversion)
        effective = [matrix for rows in self.cov_Pk_kms.values() for matrix in rows] + [matrix for matrix in self.full_cov_Pk_kms.values() if matrix is not None]
        for matrix in effective:
            tolerance = 1e-12 * max(np.max(np.abs(matrix)), np.finfo(float).tiny)
            if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, rtol=0, atol=tolerance):
                raise ValueError("effective covariance must be finite and symmetric")
        self.request = PredictionRequest(tuple(
            PredictionGroup(key, self.Rebin_data.zs[key], self.Rebin_data.k_kms[key],
                            k_coverage_ikms=[np.max(k) for k in data[key].k_kms])
            for key in data), configuration_id)

    def get_prediction_request(self):
        """Return the immutable raw-P1D prediction contract.

        Returns
        -------
        PredictionRequest
            Requested pre-contaminant, pre-rebinning grids and identity.
        """
        return self.request

    def apply_observation_model(self, p1d_lya_kms, context, nuisance_parameters, metal_moments=None):
        """Apply responses and rebin raw external predictions onto data grids.

        Parameters
        ----------
        p1d_lya_kms : mapping
            Raw P1D rows keyed by request dataset identifier.
        context : PredictionContext
            Request identity, mean flux, and velocity conversions.
        nuisance_parameters : mapping
            Likelihood nuisance values used by response models.
        metal_moments : mapping, optional
            Metal moments keyed like ``p1d_lya_kms`` (from
            :func:`project_metal_moments`), required by the Kaiser metal model.

        Returns
        -------
        dict
            Observed-model P1D rows keyed by dataset identifier.

        Raises
        ------
        ValueError
            If request identity, dataset keys, row counts, or row shapes do
            not satisfy the prediction contract.
        """
        if context.request_id != self.request.identity or set(p1d_lya_kms) != set(self.data):
            raise ValueError("prediction request identity or dataset mismatch")
        index = {z: i for i, z in enumerate(context.redshifts)}
        groups = {group.identifier: group for group in self.request.groups}
        result = {}
        for keys in self.Rebin_data.groups:
            group = groups[keys[0]]  # data sets of a group share redshifts and fine grids
            for key in keys:
                if len(p1d_lya_kms[key]) != len(groups[key].redshifts):
                    raise ValueError("redshift prediction count mismatch")
            rows = []
            for k, power in zip(group.k_ikms, p1d_lya_kms[group.identifier]):
                power = np.asarray(power, dtype=float)
                if power.shape != (len(k),) or not np.all(np.isfinite(power)):
                    raise ValueError("malformed or non-finite external prediction")
                rows.append(power)
            zs = np.asarray(group.redshifts)
            k_rows = [np.asarray(k) for k in group.k_ikms]
            inds = [index[z] for z in group.redshifts]
            mean_flux = np.asarray(context.mean_flux)[inds]
            M_of_z = np.asarray(context.dkms_diMpc)[inds]
            moments = None if metal_moments is None else metal_moments[group.identifier]
            # contaminants once per group; each data set adds its resolution correction
            base = nuisance_parameters if len(keys) == 1 else without_resolution(nuisance_parameters)
            observed, _ = apply_observation_model(self.model_cont, self.model_syst, zs, k_rows, rows, mean_flux, M_of_z, dataset_nuisance(keys[0], base, self.syst_data_labels), metal_moments=moments)
            for key in keys:
                own = observed if len(keys) == 1 else apply_resolution(
                    self.model_syst, zs, k_rows, observed,
                    dataset_nuisance(key, nuisance_parameters, self.syst_data_labels))
                result[key] = self.Rebin_data.rebinning(key, own)
        return result

    def evaluate_from_p1d(self, p1d_lya_kms, context, nuisance_parameters, metal_moments=None):
        """Evaluate correlated Gaussian likelihood for raw external P1D rows.

        Parameters
        ----------
        p1d_lya_kms : mapping
            Raw P1D rows satisfying :meth:`get_prediction_request`.
        context : PredictionContext
            Mean-flux and conversion context for the request.
        nuisance_parameters : mapping
            Nuisance values used by observational responses.
        metal_moments : mapping, optional
            Metal moments for the Kaiser metal model.

        Returns
        -------
        LikelihoodResult
            Log likelihood, data chi-squared, optional log determinant,
            number of data points, and validity flag.
        """
        predictions = self.apply_observation_model(p1d_lya_kms, context, nuisance_parameters, metal_moments)
        chi2, logdet, ndata = 0.0, 0.0, 0
        chi2_by_dataset, ndata_by_dataset = {}, {}
        for key, dataset in self.data.items():
            if dataset.full_Pk_kms is not None:
                diffs, factors = [dataset.full_Pk_kms - np.concatenate(predictions[key])], [self.full_chol_Pk_kms[key]]
            else:
                diffs, factors = [data - model for data, model in zip(dataset.Pk_kms, predictions[key])], self.chol_Pk_kms[key]
            chi2_by_dataset[key], ndata_by_dataset[key] = 0.0, 0
            for diff, factor in zip(diffs, factors):
                contribution, determinant = gaussian_residual(diff, factor)
                chi2 += contribution
                logdet += determinant if self.include_logdet else 0.0
                ndata += len(diff)
                chi2_by_dataset[key] += contribution
                ndata_by_dataset[key] += len(diff)
        return LikelihoodResult(-0.5 * (chi2 + logdet), chi2, logdet, ndata, True,
                                chi2_by_dataset, ndata_by_dataset)
