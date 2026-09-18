Installation
============

cup1d requires Python 3.12 or newer and an installation of `LaCE
<https://github.com/igmhub/LaCE>`_ for emulator and cosmology support. From a
checkout, install cup1d and its runtime dependencies with:

.. code-block:: console

   python -m pip install -e .

For development, install the test and documentation extras:

.. code-block:: console

   python -m pip install -e ".[test,docs]"

Run the regression suite with:

.. code-block:: console

   pytest -q

Build the documentation locally with:

.. code-block:: console

   make docs

The generated site is written to ``docs/_build/html/index.html``.

NERSC users
-----------

MPI analyses require an MPI-compatible ``mpi4py`` installation. Follow the
NERSC guidance for building ``mpi4py`` in the environment that runs cup1d.
