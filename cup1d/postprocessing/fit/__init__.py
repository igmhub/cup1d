"""Quantities derived from a fit (a point or a chain), computed apart from the figures.

A single best-fitting point is treated as a chain of length one, so every
function here works for both.

Modules
-------
igm
    IGM histories at a point or for a chain, the prior envelope and the
    summary table of the DR1 paper.
illustrate
    Residuals of the data after switching off contaminant terms (Fig. 7).

The contamination of metals and HCDs is computed by
:mod:`cup1d.postprocessing.posterior.contaminants`.
"""
