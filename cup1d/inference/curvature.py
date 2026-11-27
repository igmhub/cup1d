"""Gauss--Newton curvature of a P1D likelihood, shared by cup1d and lya_interface.

The local error estimate is ``H = sum_b J_b^T C_b^-1 J_b`` (one term per
independent Gaussian block ``b`` of the data) plus the prior curvature, in
unit-cube coordinates. This module holds the pieces every caller needs:

* :func:`p1d_blocks` and :func:`flatten_p1d`: split a P1D prediction into the
  same covariance blocks the likelihood uses (the full covariance of a data
  set when it has one, so cross-redshift correlations are kept; otherwise
  per-redshift blocks);
* :func:`finite_difference_jacobian`: central differences, inward one-sided
  ones at the cube edges;
* :func:`gauss_newton_curvature`: ``sum J^T C^-1 J``;
* :data:`FLAT_PRIOR_CURVATURE`: flat priors as Gaussians of the same variance.

:meth:`cup1d.inference.fitter.Fitter.estimate_mle_errors` (native fits) and
``lya_interface.diagnostics.estimate_local_errors`` (Cobaya graphs, which add
Vega blocks and Cobaya priors) both build on them.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve

# A uniform prior on [0, 1] (unit cube) has variance 1/12. A Gaussian
# approximation cannot truncate at prior edges; adding this curvature keeps
# data-unconstrained directions within their priors (DESI DR1:
# sigma(Delta2_star) 0.035 with it, 0.10 without; the posterior gives 0.032).
FLAT_PRIOR_CURVATURE = 12.0


@dataclass(frozen=True)
class P1DBlock:
    """One independent Gaussian block of a P1D likelihood.

    ``key``: data set; ``indices``: its redshift bins in the block (in order);
    ``factor``: Cholesky factor (``scipy.linalg.cho_factor``) of the block
    covariance.
    """

    key: str
    indices: tuple
    factor: tuple

    def solve(self, matrix):
        """``C^-1 @ matrix``."""
        return cho_solve(self.factor, matrix, check_finite=False)


def p1d_blocks(likelihood, zmask=None):
    """Covariance blocks of a cup1d P1D likelihood, as it evaluates chi2.

    Parameters
    ----------
    likelihood : object
        ``cup1d.likelihood.likelihood.Likelihood`` or
        ``cup1d.likelihood.external.ExternalP1DLikelihood``: exposes ``data``
        and the Cholesky factors ``chol_Pk_kms[key][iz]`` and
        ``full_chol_Pk_kms[key]``.
    zmask : array-like, optional
        Redshifts to keep (absolute tolerance 1e-3). Selecting redshifts uses
        per-redshift blocks, as the likelihood does.

    Returns
    -------
    list of P1DBlock
    """
    blocks = []
    full_factors = getattr(likelihood, "full_chol_Pk_kms", {}) or {}
    for key, data in likelihood.data.items():
        z = np.asarray(data.z)
        if zmask is None:
            indices = np.arange(len(z))
        else:
            indices = np.flatnonzero(np.any(np.isclose(z[:, None], np.atleast_1d(zmask)[None, :],
                                                       atol=1.0e-3, rtol=0), axis=1))
        if len(indices) == 0:
            continue
        full = full_factors.get(key)
        if zmask is None and full is not None and getattr(data, "full_Pk_kms", None) is not None:
            blocks.append(P1DBlock(key, tuple(int(i) for i in indices), full))
        else:
            blocks += [P1DBlock(key, (int(i),), likelihood.chol_Pk_kms[key][i]) for i in indices]
    if not blocks:
        raise ValueError("no P1D data selected")
    return blocks


def flatten_p1d(predictions, blocks):
    """One model vector per block from ``{key: [P1D of each redshift]}``."""
    return [np.concatenate([np.asarray(predictions[block.key][i]).reshape(-1) for i in block.indices])
            for block in blocks]


def difference_step(center, step):
    """Signed step for a derivative at ``center`` in ``[0, 1]``.

    ``step`` when ``center +- step`` are both in the cube (central
    difference), else a full step towards the interior (one-sided): a point
    at an edge never gets a tiny, noisy step.
    """
    if center - step >= 0.0 and center + step <= 1.0:
        return step
    return step if center + step <= 1.0 else -step


def finite_difference_jacobian(vectors, center, step):
    """Derivatives of a list of model vectors with respect to unit-cube coordinates.

    Parameters
    ----------
    vectors : callable
        ``vectors(x)`` returns a list of 1D arrays (one per data block).
    center : ndarray
        Expansion point in the unit cube.
    step : float
        Displacement; central differences where both ``center +- step`` are in
        the cube, inward one-sided differences at an edge.

    Returns
    -------
    list of ndarray
        One ``(len(block), len(center))`` Jacobian per block.
    """
    center = np.asarray(center, dtype=float)
    base = vectors(center)
    jacobians = [np.empty((len(b), center.size)) for b in base]
    for index in range(center.size):
        h = difference_step(center[index], step)
        direction = np.zeros(center.size)
        direction[index] = h
        if h == step and center[index] - step >= 0.0:
            plus, minus = vectors(center + direction), vectors(center - direction)
            columns = [(p - m) / (2 * h) for p, m in zip(plus, minus)]
        else:
            other = vectors(center + direction)
            columns = [(o - b) / h for o, b in zip(other, base)]
        for jacobian, column in zip(jacobians, columns):
            jacobian[:, index] = column
    return jacobians


def gauss_newton_curvature(jacobians, solvers):
    """``sum_b J_b^T C_b^-1 J_b``; ``solvers[b](M)`` returns ``C_b^-1 @ M``."""
    return sum(j.T @ solve(j) for j, solve in zip(jacobians, solvers))


def p1d_gauss_newton(likelihood, predict, center, step, zmask=None):
    """Gauss--Newton curvature of a P1D likelihood alone (no priors).

    ``predict(x)`` returns ``{key: [P1D of each redshift]}`` at unit-cube
    point ``x`` (rebinned to the data, with every contaminant and systematic).
    """
    blocks = p1d_blocks(likelihood, zmask=zmask)
    jacobians = finite_difference_jacobian(lambda x: flatten_p1d(predict(x), blocks), center, step)
    return gauss_newton_curvature(jacobians, [block.solve for block in blocks])
