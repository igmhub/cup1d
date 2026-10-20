"""External, pre-contaminant P1D likelihood boundary."""

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np
from scipy.linalg import cho_solve

from cup1d.utils.rebinning import Rebinning


def make_integrator(projection=None):
    """Build the P3D-to-P1D quadrature from projection settings.

    Parameters
    ----------
    projection : mapping, optional
        ``rule: k_mu`` selects ForestFlow's split Gauss--Legendre rule in ln k
        at fixed k_par (``P1DIntegratorKMu``, also cup1d's native default);
        any other ``rule`` (default ``k_perp``) the Simpson/Gauss--Legendre rule
        in ln k_perp (``P1DIntegrator``). Remaining keys are forwarded.

    Returns
    -------
    P1DIntegratorKMu or P1DIntegrator
    """
    from forestflow.statistics.p1d import P1DIntegrator, P1DIntegratorKMu

    projection = {} if projection is None else dict(projection)
    rule = projection.pop("rule", "k_perp")
    if rule == "k_mu":
        return P1DIntegratorKMu(**projection)
    if rule != "k_perp":
        raise ValueError("projection rule must be 'k_mu' or 'k_perp'")
    return P1DIntegrator(**projection)


def _k_perp_range(integrator):
    """Return the transverse integration limits of a projection quadrature."""
    if hasattr(integrator, "k_perp_iMpc"):
        return integrator.k_perp_iMpc[0], integrator.k_perp_iMpc[-1]
    return integrator.k_perp_min_iMpc, integrator.k_perp_max_iMpc


def _quadrature(integrator, k_iMpc):
    """Return ``k``, ``mu`` and an integrate function on the quadrature grid.

    ``integrate(f)`` returns ``(1/2pi) int dk_perp k_perp f`` for values ``f``
    sampled on the returned ``(k, mu)`` grid, for either quadrature rule.
    """
    if hasattr(integrator, "k_perp_iMpc"):
        k_par, k_perp = integrator.geometry(k_iMpc)
        k3d = np.hypot(k_par, k_perp)
        return k3d, k_par / k3d, integrator.integrate
    k3d, mu, weights = integrator.geometry(k_iMpc)
    return k3d, mu, lambda values: integrator.integrate(values, k3d, weights)


METAL_MOMENT_NAMES = ("A0", "A1", "I0", "I1", "I2")


def _immutable(values):
    """Return a read-only float64 copy (bytes-backed, WRITEABLE cannot be re-enabled)."""
    values = np.asarray(values, dtype=np.float64)
    return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)


def _interpolate_signed(k_model, values, k_out):
    """Cubic spline in (ln k, ln|Q|) for rows of one sign, as cup1d's P1D interpolation."""
    from cup1d.theory.theory import interpolate_p1d

    values = np.asarray(values, dtype=float)
    if np.all(values < 0):
        return -interpolate_p1d(k_model, -values, k_out)
    return interpolate_p1d(k_model, values, k_out)


def project_fused(request, snapshot, arinyo, projection=None, kmax_iMpc=np.inf,
                  integrator=None, *, model=None, linear=None, moments=False, n_k=None):
    """Project the Lya P1D and, optionally, the Kaiser metal moments in one pass.

    The linear power, ForestFlow's nonlinear correction and the Kaiser factors
    are evaluated once on the quadrature grid and shared by the Lya P1D and the
    five metal moments (see :func:`project_metal_moments`).

    Parameters
    ----------
    request, snapshot, arinyo, projection, kmax_iMpc, integrator, model, linear
        As in :func:`project_arinyo_request`.
    moments : bool, default=False
        Also return the metal moments.
    n_k : int, optional
        Project on ``n_k`` log-spaced k per requested row (spanning it) and
        interpolate to the requested grid with a cubic spline in
        ``(ln k, ln |Q|)``, as cup1d's ``n_k_emu`` does for the emulator P1D
        (:func:`cup1d.theory.theory.interpolate_p1d`). The projected quantities
        are smooth in k; metal oscillations are applied afterwards on the
        requested grid. ``None`` projects at every requested k.

    Returns
    -------
    p1d : mapping
        As :func:`project_arinyo_request`.
    metal_moments : mapping or None
        As :func:`project_metal_moments`, or ``None`` if ``moments`` is false.

    Raises
    ------
    ValueError
        If a grid exceeds declared coverage, the P1D is non-finite or
        non-positive, or a moment is non-finite.
    """
    from forestflow.model.arinyo import ArinyoModel
    from forestflow.model.linear import LinearTheoryGrid
    from forestflow.statistics.p1d import P1D_Mpc

    keys = list(dict.fromkeys((z, k) for group in request.groups
                              for z, k in zip(group.redshifts, group.k_ikms)))
    if not keys:
        raise ValueError("external coefficient projection requires P1D grids")
    if n_k is not None and (int(n_k) != n_k or n_k < 4):
        raise ValueError("n_k must be an integer >= 4 or None")
    zs = np.asarray([z for z, _ in keys], dtype=float)
    # Evaluation grids: the requested rows, or n_k log-spaced points spanning them.
    rows_kms = [np.asarray(k, dtype=float) for _, k in keys]
    eval_kms = rows_kms if n_k is None else [np.geomspace(k[0], k[-1], int(n_k)) for k in rows_kms]
    lengths = np.asarray([len(k) for k in eval_kms], dtype=int)
    conversion = np.asarray(snapshot.get_dkms_diMpc(zs), dtype=float)
    k_iMpc = np.empty((len(keys), int(lengths.max())))
    for index, k in enumerate(eval_kms):
        row = k * conversion[index]
        if row.max() > kmax_iMpc:
            raise ValueError("parallel k outside declared external-coefficient coverage")
        k_iMpc[index, :len(row)] = row
        k_iMpc[index, len(row):] = row[-1]
    integrator = make_integrator(projection) if integrator is None else integrator
    for cutoff in _k_perp_range(integrator):
        snapshot.validate_k(np.sqrt(k_iMpc**2 + cutoff**2))
    names = next(iter(arinyo.values())).keys()
    parameters = {name: np.asarray([arinyo[z][name] for z in zs])[:, None, None]
                  for name in names}
    model = ArinyoModel(fiducial_cosmology=snapshot) if model is None else model
    linear = LinearTheoryGrid(z=zs, cosmology=snapshot) if linear is None else linear

    projected_moments = None
    if hasattr(model, "nonlinear_correction"):
        # One pass: shared P_L, D_NL and Kaiser factors on the quadrature grid.
        parameters = model.default_params | parameters
        k3d, mu, integrate = _quadrature(integrator, k_iMpc)
        mu2 = mu**2
        lin_p = model.linear.get_linP_Mpc(linear, zs, k3d)
        growth = np.asarray(model.linear.get_fz(linear, zs), dtype=float)[:, None, None]
        dnl = model.nonlinear_correction(lin_p, k3d, mu, parameters)
        lya_kaiser = parameters["bias"] + parameters["bias_eta"] * growth * mu2
        projected = integrate(lin_p * lya_kaiser**2 * dnl)
        if moments:
            metal_leg = np.exp(-0.5 * (k3d / parameters["kp"]) ** 2)
            cross = lin_p * np.sqrt(dnl) * metal_leg * lya_kaiser
            metal = lin_p * metal_leg**2
            projected_moments = {
                "A0": integrate(cross),
                "A1": integrate(cross * mu2),
                "I0": integrate(metal),
                "I1": integrate(metal * mu2),
                "I2": integrate(metal * mu2**2),
            }
    else:  # minimal compatibility models only expose the Cartesian P3D
        if moments:
            raise ValueError("metal moments need a model with nonlinear_correction")
        projected = P1D_Mpc(linear, zs, k_iMpc, model.P3D_Mpc_kpar_kperp, parameters,
                            integrator=integrator)
    projected = projected * conversion[:, None]
    if projected_moments is not None:
        for name, values in projected_moments.items():
            projected_moments[name] = values * conversion[:, None]

    p1d_by_key, moments_by_key = {}, {}
    for index, key in enumerate(keys):
        row = projected[index, :lengths[index]]
        if n_k is not None:
            row = _interpolate_signed(eval_kms[index], row, rows_kms[index])
        if not np.all(np.isfinite(row)) or np.any(row <= 0):
            raise ValueError("non-finite/nonpositive external-coefficient projection")
        p1d_by_key[key] = _immutable(row)
        if projected_moments is not None:
            out = {}
            for name, values in projected_moments.items():
                moment = values[index, :lengths[index]]
                if n_k is not None:
                    moment = _interpolate_signed(eval_kms[index], moment, rows_kms[index])
                if not np.all(np.isfinite(moment)):
                    raise ValueError(f"non-finite metal moment {name}")
                out[name] = _immutable(moment)
            moments_by_key[key] = MappingProxyType(out)

    def by_group(table):
        return MappingProxyType({group.identifier: tuple(table[(z, k)]
                                 for z, k in zip(group.redshifts, group.k_ikms))
                                 for group in request.groups})

    return by_group(p1d_by_key), (by_group(moments_by_key) if moments else None)


def project_arinyo_request(request, snapshot, arinyo, projection=None,
                           kmax_iMpc=np.inf, integrator=None, *,
                           model=None, linear=None, n_k=None):
    """Project externally supplied ForestFlow coefficients to raw P1D.

    Parameters
    ----------
    request : PredictionRequest
        Immutable data-grid contract in velocity units.
    snapshot : object
        Point-specific cosmology exposing ``get_dkms_diMpc`` and
        ``validate_k``.
    arinyo : mapping
        Redshift-indexed named ForestFlow Arinyo coefficients. No emulator is
        constructed or evaluated by this function.
    projection : mapping, optional
        Projection settings for :func:`make_integrator`.
    kmax_iMpc : float, default=numpy.inf
        Largest supported parallel wavenumber in 1/Mpc.
    integrator : P1DIntegratorKMu or P1DIntegrator, optional
        Existing projection integrator. Supplying it retains its bounded
        geometry cache; otherwise one is built from ``projection``.
    model : object, optional
        Existing ForestFlow ``ArinyoModel`` (or an object exposing only
        ``P3D_Mpc_kpar_kperp``, for compatibility).
    linear : object, optional
        Matching linear-theory adapter used with ``model``.  When omitted, a
        point-specific adapter is constructed from ``snapshot``.
    n_k : int, optional
        Interpolation grid size, see :func:`project_fused`.

    Returns
    -------
    mapping
        Immutable dataset-indexed raw P1D arrays in km/s units, preserving
        ragged requested grids and their original group/redshift order.

    Raises
    ------
    ValueError
        If a requested grid exceeds declared coverage or projected power is
        non-finite/non-positive.

    Notes
    -----
    ForestFlow is imported lazily, keeping ordinary cup1d analyses independent
    of the optional external-coefficient route.
    """
    return project_fused(request, snapshot, arinyo, projection, kmax_iMpc, integrator,
                         model=model, linear=linear, n_k=n_k)[0]


def project_metal_moments(request, snapshot, arinyo, projection=None,
                          kmax_iMpc=np.inf, integrator=None, *,
                          model=None, linear=None, n_k=None):
    """Project the k_perp moments that carry Kaiser metal terms to P1D.

    For a line-of-sight Kaiser metal model the metal contributions to P1D are
    polynomials in the metal ``beta`` with these moments as coefficients (see
    :class:`cup1d.models.contaminants.metals.si_kaiser.SiKaiser`). With
    ``mu = k_par / k`` and ``(1/2pi) int dk_perp k_perp``:

    - ``A_j = (1/2pi) int dk_perp k_perp P_L sqrt(D_NL) d_m (b + b_eta f mu^2) mu^(2j)``,
      ``j = 0, 1``: Lya x metal, with ForestFlow's Lya Kaiser factor and the
      square root of its nonlinear correction (the Lya leg).
    - ``I_j = (1/2pi) int dk_perp k_perp P_L d_m^2 mu^(2j)``, ``j = 0, 1, 2``:
      metal x metal.

    ``d_m = exp(-(k/k_p)^2 / 2)`` is the pressure smoothing of one metal leg,
    with ForestFlow's Arinyo ``k_p`` (metals trace the same gas as Lya, whose
    ``D_NL`` carries ``exp(-(k/k_p)^2)`` for its two legs).

    Parameters
    ----------
    request, snapshot, arinyo, projection, kmax_iMpc, integrator, model, linear, n_k
        As in :func:`project_arinyo_request`; the same quadrature geometry is
        used, so moments and the Lya P1D share their numerical integration.
        Use :func:`project_fused` to obtain both in a single pass.

    Returns
    -------
    mapping
        Immutable dataset-indexed tuples, one per requested redshift, of
        dictionaries with keys :data:`METAL_MOMENT_NAMES`. Values are arrays on
        the requested k grid in km/s units (multiplied by ``dkms/dMpc``, like
        the P1D).

    Raises
    ------
    ValueError
        If a requested grid exceeds declared coverage or a moment is
        non-finite.
    """
    return project_fused(request, snapshot, arinyo, projection, kmax_iMpc, integrator,
                         model=model, linear=linear, moments=True, n_k=n_k)[1]


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


def apply_observation_model(model_cont, model_syst, zs, k_kms, p1d_kms, mean_flux, M_of_z, like_params=None, remove=None, metal_moments=None):
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
    metal_moments : sequence of mapping, optional
        Per-redshift moments from :func:`project_metal_moments`, required by
        the Kaiser metal model.

    Returns
    -------
    powers : list of ndarray
        Observed-model P1D rows.
    terms : list of dict
        Per-redshift response components for diagnostics.
    """
    like_params = {} if like_params is None else like_params
    syst = model_syst.get_contamination(zs, k_kms, like_params=like_params) if any(name.startswith("R_coeff") for name in like_params) else np.ones(len(zs))
    cont = model_cont.get_contamination(zs, k_kms, mean_flux, M_of_z, like_params=like_params, remove=remove, metal_moments=metal_moments)
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
            moments = None if metal_moments is None else metal_moments[group.identifier]
            observed, _ = apply_observation_model(self.model_cont, self.model_syst, np.asarray(group.redshifts), [np.asarray(k) for k in group.k_ikms], rows, np.asarray(context.mean_flux)[inds], np.asarray(context.dkms_diMpc)[inds], nuisance_parameters, metal_moments=moments)
            result[group.identifier] = self.Rebin_data.rebinning(group.identifier, observed)
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
