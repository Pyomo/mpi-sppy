SMPS Format Support
===================

The `SMPS format <https://en.wikipedia.org/wiki/SMPS_(format)>`_ is a standard
file format for stochastic programming problems. mpi-sppy can read SMPS
instances directly and solve them using the cylinder system.

Currently, only two-stage problems with ``SCENARIOS DISCRETE`` stochastic
data are supported.

SMPS Files
^^^^^^^^^^

An SMPS instance consists of three files in a directory:

- ``.cor`` -- the core deterministic model in MPS format (the "base" scenario)
- ``.tim`` -- time/stage structure mapping variables and constraints to stages
- ``.sto`` -- stochastic data defining scenarios, probabilities, and modifications

Usage
^^^^^

Use ``generic_cylinders.py`` with ``--smps-dir`` as the first argument,
pointing to the directory containing the three SMPS files. The appropriate
module (``mpisppy.problem_io.smps_module``) is inferred automatically, so
``--module-name`` is not needed:

.. code-block:: bash

   python -m mpisppy.generic_cylinders --smps-dir path/to/smps/dir \
       --solver-name cplex --EF

For example, using the sizes problem included in the repository:

.. code-block:: bash

   python -m mpisppy.generic_cylinders --smps-dir examples/sizes/SMPS \
       --solver-name cplex --EF

.. note::
   Using ``--module-name`` together with ``--smps-dir`` is an error.

The sizes SMPS example is a two-stage mixed-integer stochastic program with
10 equiprobable scenarios.

Limitations
^^^^^^^^^^^

- Only two-stage problems are supported (the ``.tim`` file must define
  exactly two stages).
- Only ``SCENARIOS DISCRETE`` format in the ``.sto`` file is supported.
  The ``INDEP`` and ``BLOCKS`` stochastic data formats are not yet implemented.
- RHS, coefficient matrix, and bounds modifications are all supported
  in scenario definitions.
- The per-stage cost expression is not separated (set to 0.0), so PH
  convergence reports may be affected.
