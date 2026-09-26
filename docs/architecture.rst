How cup1d works
===============

The :class:`cup1d.inference.analysis.Analysis` object assembles a complete
analysis from configuration, data, an emulator, and the physical models. It
then exposes the three objects most users interact with:

``analysis.data``
   One or more P1D measurements, including redshift and wavenumber grids,
   measured power, and covariance matrices.

``analysis.like``
   The likelihood that compares those measurements with predictions from
   ``analysis.theory``.

``analysis.fitter``
   The inference interface used by minimizers and samplers. It owns the
   conversion between optimizer coordinates and physical parameter
   dictionaries.

Package data flow
-----------------

The solid arrows below show construction or data flow. The dashed arrows are
the repeated evaluation loop used during minimization and sampling.

.. graphviz::

   digraph cup1d_architecture {
       graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.45", ranksep="0.65"];
       node [shape=box, style="rounded,filled", fillcolor="#eef4fb", color="#426b8a", fontname="sans-serif"];
       edge [color="#4d5963", fontname="sans-serif", fontsize=10];

       Args [label="Args / YAML\nconfiguration", fillcolor="#fff2cc"];
       Analysis [label="Analysis\norchestrator", fillcolor="#d9ead3"];
       Data [label="P1D data\nz, k, P1D, covariance"];
       Emulator [label="LaCE / ForestFlow\nemulator"];
       Theory [label="Theory\nP1D prediction"];
       Models [label="Physical models\nIGM + contaminants\n+ systematics"];
       Likelihood [label="Likelihood\nlog L + priors"];
       Fitter [label="Fitter\ncoordinate conversion\n+ inference state", fillcolor="#d9ead3"];
       Minimizer [label="Minimizer\nMLE"];
       Sampler [label="Sampler\nposterior chain"];
       Results [label="Saved results\nchains, MLE, covariance, blobs"];
       Post [label="Post-processing\nplots, tables, diagnostics", fillcolor="#eadcf8"];

       Args -> Analysis;
       Analysis -> Data [label="set_p1d"];
       Analysis -> Emulator [label="set_emulator"];
       Analysis -> Theory [label="set_theory"];
       Emulator -> Theory;
       Models -> Theory;
       Data -> Likelihood [label="measurements"];
       Theory -> Likelihood [label="prediction interface"];
       Analysis -> Likelihood [label="constructs"];
       Likelihood -> Fitter [label="constructs with"];
       Fitter -> Minimizer [label="unit-cube point"];
       Fitter -> Sampler [label="unit-cube point"];
       Minimizer -> Fitter [label="candidate point", style=dashed];
       Sampler -> Fitter [label="candidate point", style=dashed];
       Fitter -> Likelihood [label="physical parameter dict", style=dashed];
       Likelihood -> Theory [label="parameters + data grid", style=dashed];
       Theory -> Likelihood [label="model P1D + blobs", style=dashed];
       Likelihood -> Fitter [label="log posterior / chi2", style=dashed];
       Fitter -> Results;
       Results -> Post;
       Data -> Post;
       Theory -> Post;
   }

Construction phase
------------------

Creating ``Analysis(args)`` performs the following setup:

1. :class:`cup1d.configuration.Args` reads the YAML configuration and resolves
   defaults, data labels, emulator choice, free parameters, priors, and output
   settings.
2. :func:`cup1d.emulator.factory.set_emulator` creates the requested emulator.
   The emulator supplies the simulation-trained relation between cosmology,
   IGM state, and the one-dimensional flux power spectrum.
3. The P1D factory loads observational or simulated measurements. For a
   forecast or mock, a separate ``true`` theory can first generate synthetic
   data.
4. The fiducial :class:`cup1d.theory.theory.Theory` combines the emulator with
   the cosmological, IGM, contaminant, and instrumental-systematics models.
5. :class:`cup1d.likelihood.likelihood.Likelihood` receives both the data and
   theory. It prepares covariance matrices, priors, rebinning, and the
   canonical dictionaries describing free parameters.
6. :class:`cup1d.inference.fitter.Fitter` receives the likelihood and manages
   minimization, sampling, best-fit results, chains, and local error estimates.

One likelihood evaluation
-------------------------

Inference algorithms work in a unit cube because parameters can have very
different physical scales. This normalization is confined to ``Fitter``; the
likelihood and theory receive a dictionary of physical values such as
``{"As": ..., "ns": ..., "tau_eff_0": ...}``.

For every trial point:

1. ``Fitter.parameters_from_sampling_point`` converts the unit-cube array to
   the physical parameter dictionary.
2. ``Likelihood.compute_log_prob`` checks the prior bounds and evaluates any
   Gaussian priors.
3. ``Likelihood.get_p1d_kms`` asks ``Theory`` for a prediction on each data
   redshift and wavenumber grid.
4. ``Theory`` updates cosmology and the IGM models, calls the emulator, and
   applies contaminants and instrumental systematics.
5. ``Likelihood.get_log_like`` compares predicted and observed P1D using the
   appropriate per-redshift or full covariance matrix.
6. The log posterior (and optional derived ``blobs`` such as
   ``Delta2_star`` and ``n_star``) is returned through ``Fitter`` to the
   minimizer or sampler.

Minimization, sampling, and outputs
-----------------------------------

``analysis.run_minimizer`` repeatedly calls ``Fitter.minus_log_prob`` and
stores the maximum-likelihood point in both unit-cube and physical form. MLE
errors are optional and are evaluated only after minimization.

``analysis.run_sampler`` normally starts walkers around the MLE and repeatedly
calls the same likelihood path. It stores unit-cube chains, log posterior
values, and derived theory blobs. Batched LaCE calls group all
walker and redshift rows by GP expert, while covariance contractions operate
over the leading walker dimension. New blobs are stored as an ordinary float
array with shape ``(step, walker, 6)`` in the order ``Delta2_star``,
``n_star``, ``alpha_star``, ``f_star``, ``g_star``, and ``H0``.

For a vectorized sampler call, unit-cube coordinates are first converted to a
columnar parameter mapping: every parameter has shape ``(batch,)``. Theory
emulator inputs then have shape ``(batch, redshift)``; the ragged data grids
remain a list over redshift, with predictions shaped ``(batch, k_z)``.
Rebinning applies every redshift window matrix to the complete batch at once.
LaCE GP rows are flattened over ``batch * redshift`` before prediction. The
LaCE cosmology rescaling remains one inexpensive operation per point, and
legacy scalar contaminant formulae are still the next component to move onto
the batch axis. Consequently, minimization and sampling use the same priors
and model evaluation; only the inference algorithm differs.

The ``postprocessing`` package consumes these saved products to make P1D,
corner, IGM-history, contaminant, and cosmological-summary plots. Blinding is
applied to saved compressed cosmological results where configured, and
unblinding should occur only in the presentation or authorized analysis step.

Minimal entry point
-------------------

.. code-block:: python

   from cup1d import Analysis, Args

   args = Args.from_yaml("my_configuration.yaml")
   analysis = Analysis(args)

   initial = analysis.fitter.sampling_point_from_parameters()
   analysis.run_minimizer(initial)
   analysis.run_sampler()

The tutorials provide concrete versions of this workflow for observational
data, forecasts, and mocks; see :doc:`tutorials`.
