"""P1D figures: data, data against model, residuals, errors and covariance.

Modules
-------
bins
    :class:`P1DBin` and :class:`P1DPlotter` (selection of the likelihood's
    data and predictions, built in one place).
plots
    Renderers of bins: spectra, residuals, spectra with residual panels,
    standardized-error histograms; ``plot_p1d`` behind ``Likelihood.plot_p1d``.
data
    The measurement alone (``BaseDataP1D.plot_p1d``).
covariance
    Error budget (statistical, systematic, emulator), correlation matrix,
    and the systematic components of a DESI measurement file.
"""

from cup1d.postprocessing.p1d.bins import P1DBin, P1DPlotter
from cup1d.postprocessing.p1d.plots import (  # noqa: F401 - _finish and _dataset_titles for posterior.p1d
    _dataset_titles, _finish, p1d_figure_data, p1d_native_output, p1d_standardized_residuals,
    plot_p1d, plot_p1d_bins, plot_p1d_errors, plot_p1d_residuals, plot_p1d_spectra, plot_p1d_with_residuals,
)

__all__ = ["P1DBin", "P1DPlotter", "p1d_figure_data", "p1d_native_output", "p1d_standardized_residuals",
           "plot_p1d", "plot_p1d_bins", "plot_p1d_errors", "plot_p1d_residuals", "plot_p1d_spectra",
           "plot_p1d_with_residuals"]
