"""Projection of 3D models to P1D and the theory-likelihood prediction contract.

- P1D interpolation in ``(ln k, ln P)`` (:func:`interpolate_p1d`,
  :func:`interpolate_signed`);
- the fused ForestFlow/Arinyo projection of P3D (and of the Kaiser metal
  moments) to P1D on requested grids (:func:`fused_projection_Mpc`,
  :func:`project_fused`, :func:`project_metal_moments`, ...);
- the immutable request a likelihood sends to an external theory
  (:class:`PredictionRequest`, :class:`PredictionGroup`) and the context the
  theory returns with its raw P1D (:class:`PredictionContext`).
"""

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from types import MappingProxyType

import numpy as np



def interpolate_p1d(k_model, p1d_model, k_out):
    """Interpolate P1D along its last axis with a cubic spline.

    Parameters
    ----------
    k_model : array_like
        Increasing wavenumbers at which ``p1d_model`` was evaluated.
    p1d_model : array_like
        P1D values with the wavenumber axis last.
    k_out : array_like
        Wavenumbers inside ``[k_model[0], k_model[-1]]``.

    Returns
    -------
    ndarray
        Interpolated P1D. Each P1D (row along the last axis) is splined
        independently in ``(ln k, ln P)``, falling back to ``(ln k, P)`` if any
        of its values is not positive. Rows containing non-finite values, e.g.
        from an emulator extrapolating far outside its training domain, are
        returned as NaN so that the likelihood can reject that point.
    """
    from scipy.interpolate import CubicSpline

    ln_k_model = np.log(np.asarray(k_model, dtype=float))
    ln_k_out = np.log(np.asarray(k_out, dtype=float))
    p1d_model = np.asarray(p1d_model, dtype=float)
    rows = p1d_model.reshape(-1, p1d_model.shape[-1])
    out = np.full((rows.shape[0], ln_k_out.size), np.nan)
    finite = np.all(np.isfinite(rows), axis=1)
    positive = finite & np.all(rows > 0, axis=1)
    signed = finite & ~positive
    # one spline fit for all rows of each kind (independent columns)
    if np.any(positive):
        out[positive] = np.exp(CubicSpline(ln_k_model, np.log(rows[positive]), axis=1)(ln_k_out))
    if np.any(signed):
        out[signed] = CubicSpline(ln_k_model, rows[signed], axis=1)(ln_k_out)
    return out.reshape(p1d_model.shape[:-1] + (ln_k_out.size,))


def make_integrator(projection=None):
    """Build the P3D-to-P1D quadrature from projection settings.

    Parameters
    ----------
    projection : mapping, optional
        ``rule: k_mu`` (default) selects ForestFlow's split Gauss--Legendre
        rule in ln k at fixed k_par (``P1DIntegratorKMu``), the quadrature of
        native cup1d; ``rule: k_perp`` the Simpson/Gauss--Legendre rule in
        ln k_perp (``P1DIntegrator``). Remaining keys are forwarded, so an
        empty mapping gives ForestFlow's default ``P1DIntegratorKMu()``.

    Returns
    -------
    P1DIntegratorKMu or P1DIntegrator
    """
    from forestflow.statistics.p1d import P1DIntegrator, P1DIntegratorKMu

    projection = {} if projection is None else dict(projection)
    rule = projection.pop("rule", "k_mu")
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


def interpolate_signed(k_model, values, k_out):
    """Cubic spline in (ln k, ln|Q|) for rows of one sign, as cup1d's P1D interpolation.

    Rows (last axis) whose values are all negative are splined as ``-ln(-Q)``;
    the others as :func:`cup1d.theory.theory.interpolate_p1d`.
    """
    values = np.asarray(values, dtype=float)
    rows = values.reshape(-1, values.shape[-1])
    negative = np.all(rows < 0, axis=1)
    out = np.empty((rows.shape[0], np.size(k_out)))
    if np.any(negative):
        out[negative] = -interpolate_p1d(k_model, -rows[negative], k_out)
    if not np.all(negative):
        out[~negative] = interpolate_p1d(k_model, rows[~negative], k_out)
    return out.reshape(values.shape[:-1] + (np.size(k_out),))


def fused_projection_Mpc(model, linear, zs, k_iMpc, parameters, integrator, moments=False):
    """Project the Lya P1D and, optionally, the Kaiser metal moments in Mpc units.

    The shared core of :func:`project_fused` and of the native ForestFlow
    theory: the linear power, ForestFlow's nonlinear correction and the Kaiser
    factors are evaluated once on the quadrature grid.

    Parameters
    ----------
    model : forestflow.model.arinyo.ArinyoModel
        Model providing ``linear``, ``default_params``, ``kernel_terms`` and
        ``pressure_leg``.
    linear : object
        Linear-theory grid matching ``zs``.
    zs : array_like
        Redshifts, one per row of ``k_iMpc``.
    k_iMpc : ndarray
        Parallel wavenumbers in 1/Mpc, shape ``(n_z, n_k)``.
    parameters : mapping
        Arinyo coefficients with shape ``(n_z, 1, 1)`` (missing ones take the
        model defaults).
    integrator : P1DIntegratorKMu or P1DIntegrator
        Projection quadrature.
    moments : bool, default=False
        Also return the metal moments (see :func:`project_metal_moments`).

    Returns
    -------
    p1d_Mpc : ndarray
        Lya P1D in Mpc, shape ``(n_z, n_k)``.
    metal_moments_Mpc : dict or None
        Moments in Mpc keyed by :data:`METAL_MOMENT_NAMES`, or ``None``.
    """
    zs = np.asarray(zs, dtype=float)
    parameters = model.default_params | parameters
    k3d, mu, integrate = _quadrature(integrator, k_iMpc)
    mu2 = mu**2
    lin_p = model.linear.get_linP_Mpc(linear, zs, k3d)
    growth = np.asarray(model.linear.get_fz(linear, zs), dtype=float)[:, None, None]
    # ForestFlow's Arinyo kernel and its factors (shared by the metal terms)
    p3d, lya_kaiser, dnl = model.kernel_terms(lin_p, growth, k3d, mu, parameters)
    p1d = integrate(p3d)
    if not moments:
        return p1d, None
    metal_leg = model.pressure_leg(k3d, parameters["kp"])
    cross = lin_p * np.sqrt(dnl) * metal_leg * lya_kaiser
    metal = lin_p * metal_leg**2
    return p1d, {
        "A0": integrate(cross),
        "A1": integrate(cross * mu2),
        "I0": integrate(metal),
        "I1": integrate(metal * mu2),
        "I2": integrate(metal * mu2**2),
    }


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

    rows = _request_rows(request, snapshot, kmax_iMpc, n_k)
    zs, k_iMpc = rows.zs, rows.k_iMpc
    integrator = make_integrator(projection) if integrator is None else integrator
    for cutoff in _k_perp_range(integrator):
        snapshot.validate_k(np.sqrt(k_iMpc**2 + cutoff**2))
    names = next(iter(arinyo.values())).keys()
    parameters = {name: np.asarray([arinyo[z][name] for z in zs])[:, None, None]
                  for name in names}
    model = ArinyoModel(fiducial_cosmology=snapshot) if model is None else model
    linear = LinearTheoryGrid(z=zs, cosmology=snapshot) if linear is None else linear

    projected_moments = None
    if hasattr(model, "kernel_terms"):
        # One pass: shared P_L, D_NL and Kaiser factors on the quadrature grid.
        projected, projected_moments = fused_projection_Mpc(
            model, linear, zs, k_iMpc, parameters, integrator, moments=moments)
    else:  # minimal compatibility models only expose the Cartesian P3D
        if moments:
            raise ValueError("metal moments need a model with kernel_terms")
        projected = P1D_Mpc(linear, zs, k_iMpc, model.P3D_Mpc_kpar_kperp, parameters,
                            integrator=integrator)
    return _pack_request(request, rows, projected, projected_moments if moments else None)


@dataclass(frozen=True)
class _RequestRows:
    """Unique (z, k) rows of a prediction request on the evaluation grid.

    ``keys``: (z, k_ikms) of each row; ``zs``: their redshifts; ``rows_kms``:
    requested grids; ``eval_kms``: evaluation grids (the requested ones, or
    ``n_k`` log-spaced points spanning them); ``lengths``: their lengths;
    ``conversion``: dkms/dMpc per row; ``k_iMpc``: evaluation grids in 1/Mpc,
    padded with each row's last k; ``n_k`` as requested.
    """

    keys: list
    zs: np.ndarray
    rows_kms: list
    eval_kms: list
    lengths: np.ndarray
    conversion: np.ndarray
    k_iMpc: np.ndarray
    n_k: int | None


def _request_rows(request, snapshot, kmax_iMpc, n_k):
    """Unique (z, k) rows of a request, checked against the k coverage."""
    # Coverage is checked at the largest data bin centre of each row, as the
    # native kmax_ikms cut; the fine grid averaging the theory over the last
    # bin's width may extend slightly beyond it (native cup1d does not check).
    coverage = {}
    for group in request.groups:
        # duck-typed groups without coverage are checked over the whole row
        k_coverage = getattr(group, "k_coverage_ikms", None) or [row[-1] for row in group.k_ikms]
        for z, k, k_cover in zip(group.redshifts, group.k_ikms, k_coverage):
            coverage[(z, k)] = max(coverage.get((z, k), 0.0), k_cover)
    keys = list(coverage)
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
        if coverage[keys[index]] * conversion[index] > kmax_iMpc:
            raise ValueError("parallel k outside declared external-coefficient coverage")
        k_iMpc[index, :len(row)] = row
        k_iMpc[index, len(row):] = row[-1]
    return _RequestRows(keys, zs, rows_kms, eval_kms, lengths, conversion, k_iMpc, n_k)


def _pack_request(request, rows, p1d_Mpc, moments_Mpc=None):
    """P1D (and moments) in Mpc on the evaluation rows -> km/s per data set.

    Returns the P1D and, with moments, the moments, as immutable mappings
    ``{group identifier: (one array per redshift of the group)}``.
    """
    projected = p1d_Mpc * rows.conversion[:, None]
    projected_moments = None
    if moments_Mpc is not None:
        projected_moments = {name: values * rows.conversion[:, None]
                             for name, values in moments_Mpc.items()}

    p1d_by_key, moments_by_key = {}, {}
    for index, key in enumerate(rows.keys):
        row = projected[index, :rows.lengths[index]]
        if rows.n_k is not None:
            row = interpolate_signed(rows.eval_kms[index], row, rows.rows_kms[index])
        if not np.all(np.isfinite(row)) or np.any(row <= 0):
            raise ValueError("non-finite/nonpositive external-coefficient projection")
        p1d_by_key[key] = _immutable(row)
        if projected_moments is not None:
            out = {}
            for name, values in projected_moments.items():
                moment = values[index, :rows.lengths[index]]
                if rows.n_k is not None:
                    moment = interpolate_signed(rows.eval_kms[index], moment, rows.rows_kms[index])
                if not np.all(np.isfinite(moment)):
                    raise ValueError(f"non-finite metal moment {name}")
                out[name] = _immutable(moment)
            moments_by_key[key] = MappingProxyType(out)

    def by_group(table):
        return MappingProxyType({group.identifier: tuple(table[(z, k)]
                                 for z, k in zip(group.redshifts, group.k_ikms))
                                 for group in request.groups})

    return by_group(p1d_by_key), (by_group(moments_by_key) if projected_moments is not None else None)


def emulate_request(request, snapshot, emulator, inputs, kmax_iMpc=np.inf, n_k=None):
    """P1D of a prediction request from an emulator that predicts P1D directly.

    The counterpart of :func:`project_fused` for P1D emulators (LaCE GP):
    the same request rows, evaluation grids and packing, with
    ``emulator.emulate_p1d_Mpc(inputs, k_iMpc)`` instead of a P3D projection.

    Parameters
    ----------
    request : PredictionRequest
        Data-set groups with redshifts and ragged k grids in s/km.
    snapshot : object
        Cosmology exposing ``get_dkms_diMpc(z)``.
    emulator : object
        P1D emulator with ``emulate_p1d_Mpc(inputs, k_Mpc)``, ``inputs``
        mapping names to arrays with one value per row of ``k_Mpc``.
    inputs : mapping
        ``{z: {emulator input: value}}`` for every requested redshift.
    kmax_iMpc : float, default=inf
        Largest admitted parallel k (the emulator coverage).
    n_k : int, optional
        As in :func:`project_fused`.

    Returns
    -------
    mapping
        As :func:`project_fused` (P1D in km/s per data-set group).
    """
    rows = _request_rows(request, snapshot, kmax_iMpc, n_k)
    names = next(iter(inputs.values())).keys()
    row_inputs = {name: np.asarray([inputs[z][name] for z in rows.zs], dtype=float) for name in names}
    p1d_Mpc = np.asarray(emulator.emulate_p1d_Mpc(row_inputs, rows.k_iMpc), dtype=float)
    return _pack_request(request, rows, p1d_Mpc)[0]


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
        Largest supported parallel wavenumber in 1/Mpc, checked at each
        row's ``k_coverage_ikms`` (the largest data bin centre).
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
    k_coverage_ikms : iterable of float, optional
        Per row, the largest wavenumber (s/km) that must lie inside the
        model's trained coverage: the largest data bin centre. ``k_ikms``
        may extend beyond it, e.g. the fine grid that averages the theory over
        the last bin's width. ``None`` checks the whole row.
    """
    identifier: str
    redshifts: tuple
    k_ikms: tuple
    k_coverage_ikms: tuple = None

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
        if self.k_coverage_ikms is not None:
            coverage = tuple(float(k) for k in self.k_coverage_ikms)
            object.__setattr__(self, "k_coverage_ikms", coverage)
            if len(coverage) != len(self.k_ikms) or not all(np.isfinite(k) and k > 0 for k in coverage):
                raise ValueError("one finite positive coverage k is required per redshift")
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
        value = dict(identifier=self.identifier, redshifts=list(self.redshifts), k_ikms=[list(row) for row in self.k_ikms], grid_fingerprints=[fingerprint(list(row)) for row in self.k_ikms])
        if self.k_coverage_ikms is not None:
            value["k_coverage_ikms"] = list(self.k_coverage_ikms)
        return value

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

    @cached_property
    def identity(self):
        """Return a stable fingerprint of the complete prediction contract.

        Computed once: the request is immutable (it was ~40% of a Cobaya
        evaluation when recomputed at every call).

        Returns
        -------
        str
            SHA-256 hexadecimal identifier.
        """
        return fingerprint(self.to_dict())

    @cached_property
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
