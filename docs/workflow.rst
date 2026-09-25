End-to-end workflow
===================

This guide starts from YAML, constructs the data, theory, and likelihood,
evaluates physical parameters, runs inference, and reloads the saved result.
For the component diagram and repeated evaluation loop, see
:doc:`architecture`.

1. Build the analysis
---------------------

.. code-block:: python

   from cup1d import Analysis, Args

   config = "configs/cm2026/cm2026_base.yaml"
   args = Args.from_yaml(config, verbose=False)
   analysis = Analysis(args)

``Args.from_yaml`` resolves data, emulator, parameters, priors, sampler
settings, and output paths. ``Analysis`` constructs
``analysis.data``, ``analysis.theory``, ``analysis.like``, and
``analysis.fitter``.

2. Inspect data and parameters
------------------------------

.. code-block:: python

   data_label = next(iter(analysis.data))
   data = analysis.data[data_label]
   print(data.z)
   print(data.k_ikms[0].shape)
   print(data.P1D_kms[0].shape)
   print(data.cov_P1D_kms[0].shape)

   for parameter in analysis.like.free_params.values():
       print(
           parameter["name"],
           parameter["value"],
           parameter["min_value"],
           parameter["max_value"],
       )

At each redshift, wavenumber and power have shape ``(Nk,)``, while the
covariance has shape ``(Nk, Nk)``. See :doc:`conventions`.

3. Evaluate the likelihood
--------------------------

Only the fitter converts between numerical unit-cube coordinates and physical
parameter dictionaries:

.. code-block:: python

   initial_cube = analysis.fitter.sampling_point_from_parameters()
   parameters = analysis.fitter.parameters_from_sampling_point(initial_cube)
   chi2 = analysis.like.get_chi2(parameters)
   P1D_by_dataset, extra = analysis.like.get_P1D_kms(parameters)
   print(chi2)

The theory predicts ``P1D_Mpc``, converts it to ``P1D_kms``, applies
contaminants and systematics, and returns predictions on the data grid.

4. Minimize and save
--------------------

.. code-block:: python

   analysis.run_minimizer(
       initial_cube,
       estimate_errors=False,
       make_plots=False,
   )
   print(analysis.fitter.mle)
   print(analysis.fitter.mle_chi2)

``estimate_errors=False`` prevents curvature estimation inside the
optimizer. ``analysis.run_minimizer`` then saves the result automatically;
if cosmological errors are absent at that save boundary,
``save_minimizer_results`` estimates them with Gauss--Newton.
``minimizer_results.npy`` records the YAML path and fit state; YAML remains
the authoritative configuration.

5. Sample and save
------------------

.. code-block:: python

   analysis.run_sampler()

The sampler starts from the MLE unless ``pini`` is supplied. It writes
``chain.npy``, ``blobs.npy``, ``lnprob.npy``, and
``sampler_results.npy``. Sampler and standalone minimizer results remain
distinct because the best sampled point is minimized again and may differ
from the earlier fit.

6. Reload and post-process
--------------------------

.. code-block:: python

   from pathlib import Path

   minimizer_file = (
       Path(analysis.fitter.save_directory) / "minimizer_results.npy"
   )
   restored = Analysis.from_results(minimizer_file)

   from cup1d.postprocessing import Plotter

   plotter = Plotter(restored.fitter)
   plotter.plot_mle_cosmo(
       plot_errors=True,
       error_method="gauss_newton",
   )

``Analysis.from_results`` reloads the referenced YAML, rebuilds the
data/theory/likelihood graph, and restores fitter state. A sampler result also
restores its chain, blobs, and log probabilities.

Blinded compressed cosmological values remain blinded in inference state.
Unblind only for authorized presentation or comparison, as demonstrated in
``notebooks/tutorials/dr1.py``.

Use ``dr1.py`` for observations, ``forecast.py`` for forecasts,
``mock.py`` for simulations, and ``cobaya_likelihood.py`` for the
external likelihood. See :doc:`tutorials` for the full directory map.
