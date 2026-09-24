Plotting
========

Plotting implementations live in ``cup1d.postprocessing``. Scientific classes
retain short methods for compatibility, so existing calls such as
``analysis.like.plot_p1d(...)`` continue to work. Model predictions, fitting,
and likelihood evaluation remain in their original modules.

Modules are organized by subject:

* ``p1d``: likelihood spectra, residual ratios, and residual histograms.
* ``likelihood``: covariance diagnostics and re-exports of the P1D interface.
* ``igm``: model parameters, mock histories, and likelihood IGM summaries.
* ``contaminants``: contaminant parameters and their effects on spectra.
* ``inference``: minimizer best fits and parameter ellipses.
* ``data.p1d`` and ``data.simulations``: measurements and simulation diagnostics.
* ``geometry`` and ``style``: hulls, ellipses, and colormaps.

The existing ``postprocessing.Plotter`` remains the entry point for plotting a
completed fit. Its plotting calls ultimately use these implementations.
Historical code under ``old_code`` is outside the supported plotting API.

Likelihood P1D plots
--------------------

Given an initialized ``analysis`` and its sampling point ``values``:

.. code-block:: python

   from cup1d.postprocessing.p1d import plot_p1d

   # Dimensionless spectra, one axis per dataset.
   plot_p1d(analysis.like, values=values)

   # Data/model residuals, one panel per selected dataset/redshift bin.
   output = plot_p1d(
       analysis.like, values=values, residuals=True, plot_panels=True,
       zmask=[2.4, 2.8], return_all=True, show=False,
   )

   # The compatibility method accepts the same arguments.
   analysis.like.plot_p1d(values, residuals=True, plot_panels=True)

``plot_p1d`` prepares data using ``P1DPlotter`` and dispatches to either
``plot_p1d_spectra`` or ``plot_p1d_residuals``. The renderers accept prepared
``P1DBin`` objects and return a figure and the axes corresponding to each bin:

.. code-block:: python

   from cup1d.postprocessing.p1d import P1DPlotter, plot_p1d_residuals

   bins, chi2, dof = P1DPlotter(analysis.like).prepare(values, collapse=True)
   figure, axes = plot_p1d_residuals(bins)
   axes[0].set_title("My residuals")
   figure.savefig("residuals.pdf")

``return_all=True`` retains the dictionary of native data/model spectra,
errors, redshifts, chi-squared values, and probabilities keyed by dataset.
``store_data=True`` instead returns plotted coordinates: ``x0``, ``y0``,
``yerr0``, etc. For multiple datasets these dictionaries are keyed by dataset
to avoid overwriting values. ``return_all`` takes precedence over ``store_data``.
Otherwise, the high-level routine returns ``None`` as before.

``plot_fname`` saves both PDF and PNG, and ``show=False`` suppresses interactive
display. ``ylims`` accepts a single pair, one pair per visible axis, or one
pair per panel row. Single-bin and multi-dataset panel layouts are supported.

Uncertainty and compatibility
-----------------------------

Pass ``rand_posterior`` with shape ``(n_samples, n_free_parameters)`` to plot
the pointwise standard deviation of posterior predictions. Pass ``n_perturb``
to draw realizations using the likelihood's full data covariance; set
``plot_realizations=False`` to disable them.

``return_covar=True`` raises an explanatory ``NotImplementedError``: the
current likelihood does not expose rebinned emulator covariance. Previously
this option failed inside the prediction code. Posterior prediction bands
are supported, but are not an emulator-covariance estimate.

For ``z_at_time=True``, supply one sampling point per selected redshift,
in dataset order with duplicate redshifts removed. This mode cannot be
combined with joint posterior samples. Redshift masks select original
prediction indices, and fit summaries count only selected data bins.

``old_plot_p1d`` is a deprecated alias of the maintained implementation. It
emits ``DeprecationWarning``; the obsolete single-data-object renderer is
retired. ``plot_p1d_errors`` retains standardized-residual histograms and its
returned ``bins``, ``zs``, and ``(d-m)/err`` arrays.
