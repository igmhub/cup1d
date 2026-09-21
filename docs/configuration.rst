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

Configurations that are not CM2026 variations are grouped by analysis type:

``configs/forecasts``
   Synthetic forecasts, named by the covariance data set and emulator.

``configs/mocks/<simulation>``
   Simulation mocks, with separate files for each covariance and emulator
   combination.

``configs/hires``
   Joint fits that add high-resolution P1D measurements to DESI DR1.

``configs/tan2026``
   Paper-specific configurations for the TAN2026 analysis.

Load an observational configuration explicitly with:

.. code-block:: python

   args = Args.from_yaml("configs/tan2026/DESIY1_FFT3_dir_DLA_TAN.yaml")

Synthetic configurations must request the synthetic defaults:

.. code-block:: python

   forecast_args = Args.from_yaml(
       "configs/forecasts/DESIY1_QMLE3_CH24_mpgcen_gpr.yaml",
       synthetic=True,
   )
   mock_args = Args.from_yaml(
       "configs/mocks/mpg_central/DESIY1_QMLE3_CH24_mpgcen_gpr.yaml",
       synthetic=True,
   )

Multiple data sets are selected with a YAML list. For example,
``configs/hires/hires.yaml`` combines DESI DR1 with Karacayli et al. (2022):

.. code-block:: yaml

   data_label:
     - DESIY1_QMLE3
     - Karacayli2022

``scripts/data/data_sampler.py`` accepts either a CM2026 variation name or a
path to a YAML configuration file. Relative YAML paths are interpreted from
the repository root.
