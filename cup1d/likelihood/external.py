"""External, pre-contaminant P1D likelihood boundary."""

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy.linalg import cho_solve

from cup1d.utils.rebinning import Rebinning


def fingerprint(value):
    """Return a stable SHA-256 fingerprint of JSON-serializable data.

    Parameters
    ----------
    value : object
        Value accepted by :func:`json.dumps` with finite numeric values.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest of the sorted-key JSON representation.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PredictionGroup:
    """Immutable raw-P1D prediction grid for one data set.

    Parameters
    ----------
    identifier : str
        Dataset identifier.
    redshifts : iterable of float
        One redshift per requested wavenumber row.
    k_ikms : iterable of iterable of float
        Positive, strictly increasing velocity-wavenumber rows in s/km.
    """
    identifier: str
    redshifts: tuple
    k_ikms: tuple

    def __post_init__(self):
        """Normalize grids to float tuples and validate their structure.

        Raises
        ------
        ValueError
            If grids are empty, lengths differ, or a redshift/wavenumber is
            non-finite, non-positive, or not strictly increasing.
        """
        object.__setattr__(self, "redshifts", tuple(float(z) for z in self.redshifts))
        object.__setattr__(self, "k_ikms", tuple(tuple(float(k) for k in row) for row in self.k_ikms))
        if not self.redshifts or len(self.redshifts) != len(self.k_ikms):
            raise ValueError("one nonempty k grid is required per redshift")
        for z, row in zip(self.redshifts, self.k_ikms):
            k = np.asarray(row)
            if not np.isfinite(z) or k.size == 0 or np.any(~np.isfinite(k)) or np.any(k <= 0) or np.any(np.diff(k) <= 0):
                raise ValueError("prediction grids must be finite, positive and strictly increasing")

    def to_dict(self):
        """Serialize the group and per-redshift wavenumber fingerprints.

        Returns
        -------
        dict
            JSON-compatible identifier, grids, and fingerprint list.
        """
        return dict(identifier=self.identifier, redshifts=list(self.redshifts), k_ikms=[list(row) for row in self.k_ikms], grid_fingerprints=[fingerprint(list(row)) for row in self.k_ikms])

    @classmethod
    def from_dict(cls, value):
        """Build a prediction group from serialized metadata.

        Parameters
        ----------
        value : mapping
            Serialized group, optionally including grid fingerprints.

        Returns
        -------
        PredictionGroup
            Validated immutable group.

        Raises
        ------
        ValueError
            If supplied grid fingerprints do not match reconstructed grids.
        """
        fields = dict(value)
        expected = fields.pop("grid_fingerprints", None)
        group = cls(**fields)
        if expected is not None and expected != group.to_dict()["grid_fingerprints"]:
            raise ValueError("serialized grid fingerprint mismatch")
        return group


@dataclass(frozen=True)
class PredictionRequest:
    """Immutable raw-P1D contract shared by theory and likelihood.

    ``groups`` describe the exact requested data-set grids; the remaining
    fields make units and prediction stage explicit and fingerprintable.
    """
    groups: tuple
    configuration_id: str
    units: str = "s/km"
    stage: str = "uncontaminated_before_rebinning"

    def __post_init__(self):
        """Freeze groups and validate identifiers, units, and prediction stage.

        Raises
        ------
        ValueError
            If a dataset identifier is repeated or unsupported units/stage are
            requested.
        """
        object.__setattr__(self, "groups", tuple(self.groups))
        if len({group.identifier for group in self.groups}) != len(self.groups):
            raise ValueError("duplicate dataset identifier")
        if self.units != "s/km" or self.stage != "uncontaminated_before_rebinning":
            raise ValueError("unsupported prediction units/stage")

    @property
    def identity(self):
        """Return a stable fingerprint of the complete prediction contract.

        Returns
        -------
        str
            SHA-256 hexadecimal identifier.
        """
        return fingerprint(self.to_dict())

    @property
    def redshifts(self):
        """Return sorted unique redshifts requested by all dataset groups.

        Returns
        -------
        tuple of float
            Unique redshifts in ascending order.
        """
        return tuple(sorted({z for group in self.groups for z in group.redshifts}))

    def to_dict(self):
        """Serialize the prediction request to JSON-compatible metadata.

        Returns
        -------
        dict
            Groups, configuration identity, units, and prediction stage.
        """
        return dict(groups=[group.to_dict() for group in self.groups], configuration_id=self.configuration_id, units=self.units, stage=self.stage)

    @classmethod
    def from_dict(cls, value):
        """Build a prediction request from serialized metadata.

        Parameters
        ----------
        value : mapping
            Serialized request containing a ``groups`` sequence.

        Returns
        -------
        PredictionRequest
            Validated immutable request.
        """
        return cls(groups=tuple(PredictionGroup.from_dict(group) for group in value["groups"]), **{key: item for key, item in value.items() if key != "groups"})


@dataclass(frozen=True)
class PredictionContext:
    """Redshift-dependent IGM and cosmology quantities for one request.

    The values must be ordered identically by redshift and have one entry for
    each requested context redshift.
    """
    request_id: str
    redshifts: tuple
    mean_flux: tuple
    dkms_diMpc: tuple

    def __post_init__(self):
        """Normalize context vectors to tuples and validate physical values.

        Raises
        ------
        ValueError
            If lengths or redshifts are inconsistent, values are non-finite,
            mean flux lies outside ``(0, 1)``, or conversion is non-positive.
        """
        for key in ("redshifts", "mean_flux", "dkms_diMpc"):
            object.__setattr__(self, key, tuple(float(item) for item in getattr(self, key)))
        if len(set(self.redshifts)) != len(self.redshifts) or not (len(self.redshifts) == len(self.mean_flux) == len(self.dkms_diMpc)):
            raise ValueError("context requires unique, matching redshifts")
        if not np.all(np.isfinite(self.redshifts + self.mean_flux + self.dkms_diMpc)) or any(not 0 < flux < 1 for flux in self.mean_flux) or any(conversion <= 0 for conversion in self.dkms_diMpc):
            raise ValueError("invalid mean flux or velocity conversion")


@dataclass(frozen=True)
class LikelihoodResult:
    """Correlated Gaussian P1D likelihood diagnostics."""
    loglike: float
    chi2_data: float
    logdet_cov: float
    ndata: int
    valid: bool


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


def apply_observation_model(model_cont, model_syst, zs, k_kms, p1d_kms, mean_flux, M_of_z, like_params=None, remove=None):
    """Apply contaminant, resolution, and IC responses to raw P1D arrays.

    Parameters
    ----------
    model_cont, model_syst : object
        cup1d contaminant and systematic response models.
    zs, k_kms, p1d_kms : sequence
        Redshifts, matching wavenumber rows in s/km, and uncontaminated P1D
        rows.
    mean_flux, M_of_z : array-like
        Mean flux and km/s-to-Mpc conversion at each redshift.
    like_params : mapping, optional
        Nuisance values used by response models.
    remove : sequence of str, optional
        Contaminant terms to omit.

    Returns
    -------
    powers : list of ndarray
        Observed-model P1D rows.
    terms : list of dict
        Per-redshift response components for diagnostics.
    """
    like_params = {} if like_params is None else like_params
    syst = model_syst.get_contamination(zs, k_kms, like_params=like_params) if any(name.startswith("R_coeff") for name in like_params) else np.ones(len(zs))
    cont = model_cont.get_contamination(zs, k_kms, mean_flux, M_of_z, like_params=like_params, remove=remove)
    powers, terms = [], []
    for index, z in enumerate(zs):
        powers.append((cont["cont_HCD"][index] * cont["cont_mul_metals"][index] * cont["IC_corr"][index] * p1d_kms[index] + cont["cont_add_metals"][index]) * syst[index])
        terms.append(dict(z=z, k_kms=k_kms[index], p1d_emu_kms=p1d_kms[index], C_res=syst[index], C_mul_metals=cont["cont_mul_metals"][index], C_add_metals=cont["cont_add_metals"][index], C_HCD=cont["cont_HCD"][index], IC_corr=cont["IC_corr"][index], p1d_tot_kms="[(C_mul_metals * C_HCD * IC_corr * p1d_emu_kms + C_add_metals) * C_res]"))
    return powers, terms


class ExternalP1DLikelihood:
    """Evaluate static P1D data against externally supplied raw predictions.

    This boundary owns data rebinning, observational responses, and effective
    covariance.  Theory code supplies only uncontaminated P1D rows satisfying
    the immutable :class:`PredictionRequest` contract.
    """

    def __init__(self, data, model_cont, model_syst, cov_factor, emulator_covariance, fiducial_conversion, emu_cov_type="block", k_rebin_factor=1, configuration_id=None, include_logdet=False):
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
        self.Rebin_data = Rebinning(data, k_rebin_factor=k_rebin_factor)
        self.cov_factor = {key: np.asarray(value) for key, value in cov_factor.items()}
        self.emu_cov_type, self.include_logdet = emu_cov_type, include_logdet
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
        self.request = PredictionRequest(tuple(PredictionGroup(key, self.Rebin_data.zs[key], self.Rebin_data.k_kms[key]) for key in data), configuration_id)

    def get_prediction_request(self):
        """Return the immutable raw-P1D prediction contract.

        Returns
        -------
        PredictionRequest
            Requested pre-contaminant, pre-rebinning grids and identity.
        """
        return self.request

    def apply_observation_model(self, p1d_lya_kms, context, nuisance_parameters):
        """Apply responses and rebin raw external predictions onto data grids.

        Parameters
        ----------
        p1d_lya_kms : mapping
            Raw P1D rows keyed by request dataset identifier.
        context : PredictionContext
            Request identity, mean flux, and velocity conversions.
        nuisance_parameters : mapping
            Likelihood nuisance values used by response models.

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
        result = {}
        for group in self.request.groups:
            if len(p1d_lya_kms[group.identifier]) != len(group.redshifts):
                raise ValueError("redshift prediction count mismatch")
            rows = []
            for k, power in zip(group.k_ikms, p1d_lya_kms[group.identifier]):
                power = np.asarray(power, dtype=float)
                if power.shape != (len(k),) or not np.all(np.isfinite(power)):
                    raise ValueError("malformed or non-finite external prediction")
                rows.append(power)
            inds = [index[z] for z in group.redshifts]
            observed, _ = apply_observation_model(self.model_cont, self.model_syst, np.asarray(group.redshifts), [np.asarray(k) for k in group.k_ikms], rows, np.asarray(context.mean_flux)[inds], np.asarray(context.dkms_diMpc)[inds], nuisance_parameters)
            result[group.identifier] = self.Rebin_data.rebinning(group.identifier, observed)
        return result

    def evaluate_from_p1d(self, p1d_lya_kms, context, nuisance_parameters):
        """Evaluate correlated Gaussian likelihood for raw external P1D rows.

        Parameters
        ----------
        p1d_lya_kms : mapping
            Raw P1D rows satisfying :meth:`get_prediction_request`.
        context : PredictionContext
            Mean-flux and conversion context for the request.
        nuisance_parameters : mapping
            Nuisance values used by observational responses.

        Returns
        -------
        LikelihoodResult
            Log likelihood, data chi-squared, optional log determinant,
            number of data points, and validity flag.
        """
        predictions = self.apply_observation_model(p1d_lya_kms, context, nuisance_parameters)
        chi2, logdet, ndata = 0.0, 0.0, 0
        for key, dataset in self.data.items():
            if dataset.full_Pk_kms is not None:
                diffs, factors = [dataset.full_Pk_kms - np.concatenate(predictions[key])], [self.full_chol_Pk_kms[key]]
            else:
                diffs, factors = [data - model for data, model in zip(dataset.Pk_kms, predictions[key])], self.chol_Pk_kms[key]
            for diff, factor in zip(diffs, factors):
                contribution, determinant = gaussian_residual(diff, factor)
                chi2 += contribution
                logdet += determinant if self.include_logdet else 0.0
                ndata += len(diff)
        return LikelihoodResult(-0.5 * (chi2 + logdet), chi2, logdet, ndata, True)
