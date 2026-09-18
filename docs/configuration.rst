Configuration
=============

Analyses use YAML configuration layered on top of a compact canonical default
file. This keeps ordinary analyses reproducible while allowing a variation to
change only the relevant options.

CM2026 analyses
---------------

The observational defaults live in
``configs/cm2026/cm2026_defaults.yaml``. The standard baseline overlay is
``configs/cm2026/cm2026_base.yaml`` and named variations live in
``configs/cm2026/variations``.

Create arguments for the baseline or a named variation with:

.. code-block:: python

   from cup1d import Analysis, Args

   baseline_args = Args.from_baseline()
   variation_args = Args.from_variation("zmin")
   analysis = Analysis(variation_args)

Each variation YAML is sparse: its values override the baseline defaults.
Redshift nodes, covariance grids, fiducial model values, priors, and emulator
training-set choices are derived internally where appropriate.

Other analyses
--------------

Paper-specific configurations belong in their own directory, such as
``configs/tan2026``. Load one explicitly with:

.. code-block:: python

   args = Args.from_yaml("configs/tan2026/DESIY1_FFT3_dir_DLA_TAN.yaml")

``scripts/data/data_sampler.py`` accepts either a CM2026 variation name or a
path to a YAML configuration file. Relative YAML paths are interpreted from
the repository root.
