Quick Start
===========

Installation
------------

Install from source using pip in editable mode from the mpi-sppy repo root directory:

::

   pip install -e .

We recommend installing from GitHub rather than pip, because
the software is under active development and the pip version is almost
always out of date. You can also include the extras flag ``docs`` to
install documentation dependencies.


Verify Installation
-------------------

Verify that mpi-sppy and a solver are installed by running:

.. code-block:: bash

   cd examples/farmer
   python -m mpisppy.generic_cylinders --module-name farmer --help


If you intend to use decomposition (PH, APH, etc.), you also need a
working installation of MPI and ``mpi4py``; see :ref:`Install mpi4py`.
If you only need to solve the extensive form directly, MPI is not required.


What You Need to Provide
-------------------------

To use mpi-sppy, you create a Python module with the following functions:

- ``scenario_creator`` -- builds a Pyomo model for one scenario (see :ref:`scenario_creator`)
- ``scenario_names_creator`` -- returns the list of scenario names (see :ref:`helper_functions`)
- ``kw_creator`` -- returns keyword arguments for the scenario creator (see :ref:`helper_functions`)
- ``inparser_adder`` -- adds problem-specific command-line arguments (see :ref:`helper_functions`)
- ``scenario_denouement`` -- called at termination (can be ``None``; see :ref:`helper_functions`)

Once you have these functions, you can use ``generic_cylinders.py``
(see :ref:`generic_cylinders`) to solve your problem using the EF or
the hub-and-spoke system. See the ``farmer`` directory in ``examples``
for a complete working example (``farmer.py`` and ``farmer_generic.bash``).


Running the Farmer Example
---------------------------

**Solve the EF** (no MPI needed):

.. code-block:: bash

   python -m mpisppy.generic_cylinders --module-name farmer \
       --num-scens 3 --EF --EF-solver-name gurobi

**Run PH with spokes** (requires MPI):

.. code-block:: bash

   mpiexec -np 3 python -m mpi4py mpisppy/generic_cylinders.py \
       --module-name farmer --num-scens 3 \
       --solver-name gurobi_persistent --max-iterations 10 \
       --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01

For more detail, see :ref:`generic_cylinders` and :ref:`Examples`.


Researchers Who Want to Compare with mpi-sppy
----------------------------------------------

The quickest approach is to run one of the canned examples in
subdirectories of ``examples``. Sample commands can be found in
``examples.runall.py``. The mpi-sppy paper in MPC provides references
for many of the examples.
