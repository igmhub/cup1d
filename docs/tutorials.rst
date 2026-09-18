Tutorials
=========

The primary end-to-end example is
``notebooks/tutorials/dr1_tutorial.py``. It constructs the CM2026 baseline,
builds an :class:`cup1d.inference.Analysis`, evaluates the likelihood, and
shows post-processing access through the analysis object.

Notebook sources are maintained as Python files. To generate Jupyter notebooks
after installing ``jupytext``:

.. code-block:: console

   jupytext --to ipynb notebooks/*/*.py

Further examples are grouped by purpose under ``notebooks/tutorials``,
``notebooks/emulator``, ``notebooks/p1d_measurements``, and
``notebooks/validation``.
