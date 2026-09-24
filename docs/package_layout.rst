Package layout
==============

The package is organized by responsibility rather than by historical analysis
steps:

``configuration``
   YAML loading, defaults, and the :class:`cup1d.configuration.Args` object.

``p1ds``
   Observational, simulated, forecast, and mock P1D data interfaces.

``emulator``
   Emulator archive selection and emulator construction.

``theory``
   Cosmology, CAMB, linear-power, and Lyman-alpha theory construction.

``models``
   IGM, contaminant, metal, HCD, feedback, and instrumental-systematic models.

``likelihood``
   Likelihood evaluation and likelihood parameter definitions.

``inference``
   Analysis orchestration, fitting, minimization, and sampling.

``postprocessing``
   Chain handling, plotting, tables, and other derived products. All active
   plotting implementations live here, with compatibility wrappers on the
   scientific classes. See :doc:`plotting`.

``old_code``
   Preserved legacy implementations that are not part of the active API.

The main public entry points are:

.. code-block:: python

   from cup1d import Analysis, Args
