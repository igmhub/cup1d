"""IGM histories implied by a fit: at a point, for a chain, prior envelope, table."""

import numpy as np

from cup1d.postprocessing.posterior.histories import igm_histories
from cup1d.postprocessing.posterior.stats import weighted_quantile

# quantity -> key of the fiducial IGM nodes (args.fid_igm) that sets its nodes
NODE_KEYS = {"mF": "tau_eff_znodes", "tau_eff": "tau_eff_znodes", "T0": "sigT_kms_znodes",
             "sigT_kms": "sigT_kms_znodes", "gamma": "gamma_znodes", "kF_kms": "kF_kms_znodes"}


def point_columns(point):
    """``{name: value}`` -> ``{name: array of length 1}`` (a chain of length one)."""
    return {name: np.atleast_1d(np.asarray(value, dtype=float)) for name, value in point.items()}


def igm_at_point(igm, point, z):
    """IGM histories at one physical point, each of shape ``(1, nz)``."""
    return igm_histories(igm, point_columns(point), z)


def igm_prior_envelope(like, z):
    """Fiducial IGM histories and those at the lower and upper prior edges.

    Parameters
    ----------
    like : cup1d.likelihood.likelihood.Likelihood
        Likelihood with ``free_params`` and ``theory.model_igm``.
    z : array_like
        Redshifts.

    Returns
    -------
    dict
        ``z`` and ``fiducial``, ``low``, ``high`` (outputs of
        :func:`igm_at_point` at the centre and edges of the unit cube).
    """
    from cup1d.likelihood import parameter as parameter_space

    n = len(like.free_params)
    out = {"z": np.asarray(z, dtype=float)}
    for key, cube in (("fiducial", 0.5), ("low", 0.0), ("high", 1.0)):
        values = parameter_space.values_from_cube(like.free_params, np.full(n, cube))
        out[key] = igm_at_point(like.theory.model_igm, values, z)
    return out


def igm_nodes(histories, z, fid_igm, quantities=("mF", "T0", "gamma")):
    """Values of one-point histories at the redshift nodes of their parameters.

    Parameters
    ----------
    histories : mapping
        Output of :func:`igm_at_point`.
    z : array_like
        Redshifts of ``histories``.
    fid_igm : mapping
        Fiducial IGM settings (``args.fid_igm``) with the ``*_znodes``.
    quantities : sequence of str
        Quantities to evaluate.

    Returns
    -------
    dict
        ``{quantity: (z_nodes, values)}`` for the quantities with nodes.
    """
    out = {}
    for quantity in quantities:
        key = NODE_KEYS.get(quantity)
        if key in fid_igm:
            nodes = np.asarray(fid_igm[key], dtype=float)
            out[quantity] = (nodes, np.interp(nodes, z, np.asarray(histories[quantity])[0]))
    return out


def igm_histories_from_chain(igm, samples, z, n_draws=20000, seed=0):
    """IGM histories of (up to ``n_draws``) unit-weight draws of a posterior.

    Returns
    -------
    histories : dict
        Output of :func:`igm_histories`, shape ``(n_draws, nz)``.
    """
    draws = samples.subsample(n_draws, seed)
    return igm_histories(igm, draws.columns(), z)


def igm_summary_table(histories, z, quantities=("mF", "T0", "gamma"), weights=None,
                      quantiles=(0.16, 0.5, 0.84)):
    """Posterior quantiles of IGM histories at given redshifts.

    Parameters
    ----------
    histories : mapping
        Output of :func:`igm_histories` (or :func:`igm_histories_from_chain`)
        evaluated at the table redshifts ``z``.
    z : array_like
        Redshifts of the table.
    quantities : sequence of str
        Columns of the table.
    weights : array_like, optional
        Sample weights.
    quantiles : sequence of float
        Quantiles (16, 50, 84 per cent by default).

    Returns
    -------
    dict
        ``z`` and, per quantity, an array of shape ``(len(quantiles), nz)``.
    """
    out = {"z": np.asarray(z, dtype=float)}
    for quantity in quantities:
        values = np.asarray(histories[quantity])
        out[quantity] = np.array([weighted_quantile(values[:, i], quantiles, weights)
                                  for i in range(values.shape[1])]).T
    return out
