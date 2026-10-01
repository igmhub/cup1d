Tutorials
=========

The primary end-to-end example is ``notebooks/tutorials/dr1.py``. Start here:
it loads the CM2026 baseline YAML, builds an
:class:`cup1d.inference.analysis.Analysis`, evaluates the DESI DR1 likelihood,
and demonstrates fitting and post-processing through the analysis object.

Notebook organization
---------------------

``notebooks/tutorials``
   Supported introductory workflows. Alongside the main DR1 tutorial, this
   directory contains focused forecast, mock, and Cobaya examples.

``notebooks/data/observations`` and ``notebooks/data/mocks``
   Inspection and comparison of observational P1D measurements and synthetic
   data products.

``notebooks/likelihood``
   Likelihood diagnostics, initial-condition generation, compressed-parameter
   calculations, and inspection of completed fits. ``inference_methods.py``
   compares baseline Nelder--Mead, PSO, and scalar versus batched sampling.

``notebooks/igm`` and ``notebooks/contaminants``
   Focused plots of IGM histories, metal contamination, HCD models, and other
   nuisance components.

``notebooks/planck``
   Planck-chain loading, importance sampling, historical comparisons, and
   cosmological figures.

``notebooks/cosmo``
   Cosmological compression and scaling studies. ``plots_cosmo_scaling.py``
   compares changed backgrounds before and after matching star parameters,
   using LaCE cosmologies and ForestFlow's P1D projection. Figure and local
   Zenodo-data saving are opt-in.

``notebooks/figures_CM26``
   Scripts and paired notebooks used to reproduce CM2026 figures and tables.

``notebooks/wip`` and ``notebooks/old``
   Work in progress and preserved legacy material. These are not supported as
   introductory tutorials.

Notebook sources are maintained as Python files. To generate Jupyter notebooks
after installing ``jupytext``:

.. code-block:: console

   jupytext --sync notebooks/tutorials/dr1.py
