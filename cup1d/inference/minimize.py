"""Bounded Nelder--Mead minimization with restarts, shared by cup1d and lya_interface."""

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class RestartResult:
    """Outcome of :func:`nelder_mead_restarts`."""

    x: np.ndarray
    """Best point found, in the unit cube."""
    fun: float
    """Objective at ``x`` (minus the log posterior)."""
    evaluations: int
    """Objective evaluations over all restarts."""
    restarts: int
    """Nelder--Mead runs."""
    success: bool
    """Converged (SciPy success, or no further improvement), not budget-limited."""
    message: str


def nelder_mead_restarts(
    objective,
    start,
    neval=1000,
    chi2_tol=0.1,
    xatol=1e-6,
    max_restarts=None,
    log=None,
):
    """Minimize ``objective`` in the unit cube with repeated Nelder--Mead runs.

    Each run starts from the best point so far. The search stops when a run
    converges (SciPy's ``success``) or when two consecutive runs improve
    ``-2 log posterior`` (twice the objective) by less than ``chi2_tol``.

    Parameters
    ----------
    objective : callable
        ``objective(x)`` is minus the log posterior at a unit-cube point
        ``x``; non-finite values are rejected points.
    start : array_like
        Initial unit-cube point.
    neval : int, default=1000
        Maximum evaluations (and iterations) of each run.
    chi2_tol : float, default=0.1
        Tolerance on ``-2 log posterior``: SciPy's ``fatol`` (on the
        objective) and the improvement below which a restart does not count.
    xatol : float, default=1e-6
        SciPy's ``xatol``; both tolerances must be met for convergence.
    max_restarts : int, optional
        Upper limit on the number of runs (no limit by default).
    log : callable, optional
        ``log(message)`` progress reports.

    Returns
    -------
    RestartResult
    """
    best_x = np.array(start, dtype=float)
    best = float(objective(best_x))
    evaluations, restarts, repeats = 1, 0, 0
    success, message = False, "maximum number of restarts reached"
    clock = time.time()
    while max_restarts is None or restarts < max_restarts:
        result = minimize(
            objective,
            best_x,
            method="Nelder-Mead",
            bounds=((0.0, 1.0),) * len(best_x),
            options={"fatol": chi2_tol, "xatol": xatol, "maxiter": neval, "maxfev": neval},
        )
        restarts += 1
        evaluations += int(result.nfev)
        improvement = 2 * (best - float(result.fun))  # in -2 log posterior
        if np.isfinite(result.fun) and result.fun < best:
            best, best_x = float(result.fun), np.array(result.x, dtype=float)
        if log is not None:
            log(f"restart {restarts}: -2 log posterior {2 * best:.4f}, improvement "
                f"{improvement:.4f}, {evaluations} evaluations, {time.time() - clock:.1f} s")
        if result.success:
            success, message = True, "Nelder-Mead converged"
            break
        repeats = repeats + 1 if improvement <= chi2_tol else 0
        if repeats >= 2:
            # two restarts in a row without real improvement
            success, message = True, f"two restarts improved -2 log posterior by less than {chi2_tol}"
            break
    return RestartResult(best_x, best, evaluations, restarts, success, message)
