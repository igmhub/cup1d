"""Figures of a P1D posterior, independent of the sampler that produced it.

Every routine takes :class:`PosteriorSamples` (weighted samples of physical
parameters in native cup1d names) and, for derived quantities, the static
cup1d IGM and contaminant models. Native cup1d (emcee) runs are read with
:meth:`PosteriorSamples.from_cup1d_folder`; other samplers (e.g. Cobaya chains
from lya_interface) build :class:`PosteriorSamples` directly.

Modules
-------
samples
    :class:`PosteriorSamples` and parameter labels.
stats
    Weighted quantiles, correlations and density contours.
star
    Delta2_star--n_star posterior and comparison with earlier measurements.
parameters
    Parameter groups against the star parameters; correlation matrix.
histories
    IGM histories (with DESI DR1 and literature) and systematics against z.
contaminants
    Metal and HCD contamination at one redshift.
p1d
    Best-fitting P1D, one row per data set.
plotter
    :class:`PosteriorPlotter`, which writes all of them to a folder.
"""

from cup1d.postprocessing.posterior.samples import PosteriorSamples, parameter_label
from cup1d.postprocessing.posterior.plotter import PosteriorPlotter

__all__ = ["PosteriorSamples", "PosteriorPlotter", "parameter_label"]
