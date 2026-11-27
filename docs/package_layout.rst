Package layout
==============

The package is organized by responsibility rather than by historical analysis
steps:

``configuration``
   YAML loading, defaults, and the :class:`cup1d.configuration.Args` object.

``p1ds``
   P1D data interfaces: measurements in ``p1ds/observations`` and synthetic
   data (simulation mocks and forecasts) in ``p1ds/synthetic``.

``emulator``
   Emulator archive selection, emulator construction, and the training-set
   convex hull.

``theory``
   Cosmology, linear-power, projection, and Lyman-alpha theory construction.

``models``
   IGM, contaminant, metal, HCD, resolution, and observation models.

``likelihood``
   Likelihood evaluation and likelihood parameter definitions.

``inference``
   Analysis orchestration, fitting, minimization, and sampling.

``postprocessing``
   Chain handling, plotting, tables, and other derived products. All active
   plotting implementations live here, with compatibility wrappers on the
   scientific classes. See :doc:`plotting`.

``old_code``
   Only ``old_input.py``, the legacy ``Args`` used by
   ``scripts/configs/generate_cm2026.py``. Not part of the active API.

The main public entry points are:

.. code-block:: python

   from cup1d import Analysis, Args
