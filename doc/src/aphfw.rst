.. _sec-aphfw:

FW-APH (compiled hub)
=====================

FW-APH is asynchronous projective hedging (see :ref:`sec-aph`) whose
scenario subproblems are solved by Frank-Wolfe steps. It runs as a hub,
selected with ``--aphfw-hub``, and works with the usual spokes. Unlike the
other hubs, the algorithm is C++: the hub writes each scenario as an MPS
file and hands it to a compiled module, ``aphfw_cpp``, which solves the
subproblems directly with Gurobi, CPLEX or Xpress.

That module is optional, and a normal mpi-sppy install does not build it.
Nothing else in mpi-sppy depends on it: without it, every other hub and
spoke works as usual, and only ``--aphfw-hub`` stops, with a message
saying how to build it.

What you need
-------------

* **A clone of mpi-sppy from GitHub.** The C++ sources are in
  ``mpisppy/aphfw_cpp/`` of the repository; the package installed from
  PyPI does not contain them. Install mpi-sppy from the clone
  (``pip install -e .`` in its root) and build from there.
* **MPI and mpi4py** (see :ref:`Install mpi4py`). The module shares
  mpi4py's communicator, so it must be compiled against the *same* MPI that
  mpi4py uses. A mismatch is not a build error; it shows up as a crash at
  run time, which is why the build command checks for it.
* **A C++ compiler with C++23 support**, reached through that MPI's
  ``mpicxx`` wrapper.
* **One of Gurobi, CPLEX or Xpress, as a full installation** with its C or
  C++ libraries and headers. The module is built for exactly one of them.
  For Gurobi, this is the platform directory of a full install (e.g.
  ``.../gurobi1302/linux64``): the ``gurobipy`` wheel does not contain the
  C++ library. The build looks for Gurobi 13.0 or 12.0.
* **CMake 3.16 or later.** If there is no suitable CMake, pip fetches one
  during the build.

The build has been tested on Linux with GCC 14.3 (conda-forge), Open MPI
5.0.8 with mpi4py 4.1.0, Gurobi 13.0.2 and CPLEX 22.1.1. Xpress support is
in the code but has not been tested in this build, and neither has macOS
or Windows.

Building
--------

From the root of your clone, in the Python environment where mpi-sppy and
mpi4py are installed::

   python -m mpisppy.aphfw_cpp.build --solver gurobi --solver-home /path/to/gurobi1302/linux64

For CPLEX, the home is the ``CPLEX_Studio...`` directory::

   python -m mpisppy.aphfw_cpp.build --solver cplex --solver-home /opt/ibm/ILOG/CPLEX_Studio2211

Without ``--solver``, the command uses whichever of ``GUROBI_HOME``,
``CPLEX_ROOT`` or ``XPRESS_HOME`` (or ``XPRESSDIR``) is set in the
environment, and asks you to choose if more than one is set. The command:

#. checks that the solver home holds what the build needs;
#. picks the MPI C++ compiler: ``--cxx`` if given, else the ``mpicxx``
   that mpi4py recorded at its own build, else the ``mpicxx`` in the same
   directory as the running ``python`` (as in a conda environment), else
   the one on ``PATH``;
#. runs ``pip install`` on ``mpisppy/aphfw_cpp``, which compiles the
   module (about 20 seconds on the 8-core test machine) and installs it
   into the environment's ``site-packages``;
#. imports the new module in a fresh interpreter and checks that it and
   mpi4py load the same MPI library (on Linux; elsewhere it says that it
   could not check).

It prints each choice before acting on it, so ``--dry-run`` shows the
solver, the compiler and the ``pip`` command without building anything.
Arguments after ``--`` go to pip (e.g. ``-- -q``). Running the command
again rebuilds the module and replaces the installed one, e.g. to switch
solver. ``python -m mpisppy.aphfw_cpp.build --help`` lists the options.

A successful build ends with lines like::

   aphfw_cpp installed at .../site-packages/aphfw_cpp.cpython-311-x86_64-linux-gnu.so
   aphfw_cpp and mpi4py both load .../lib/libmpi.so.40.40.7

The installed module records where the solver and MPI libraries are, so it
needs no ``LD_LIBRARY_PATH``.

Building without the wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The command is a convenience over a plain pip build, which you can run
yourself. ``CXX`` must be the ``mpicxx`` of mpi4py's MPI, and the solver
and its home have to be named; there is no default solver::

   CXX=mpicxx pip install ./mpisppy/aphfw_cpp \
       -C cmake.define.USE_GUROBI=ON -C cmake.define.GUROBI_HOME=/path/to/gurobi1302/linux64

For CPLEX (Xpress likewise, with ``USE_XPRESS`` and ``XPRESS_HOME``)::

   CXX=mpicxx pip install ./mpisppy/aphfw_cpp \
       -C cmake.define.USE_CPLEX=ON -C cmake.define.CPLEX_ROOT=/opt/ibm/ILOG/CPLEX_Studio2211

This skips the checks, including the one for a mismatched MPI.

Checking the build
------------------

``python -c "import aphfw_cpp"`` should succeed from any directory. The
tests in ``mpisppy/tests/test_aphfw_hub.py`` run end to end once the module
is built, including eight MPI tests that each start their own ``mpiexec``
job (3 or 6 ranks)::

   cd mpisppy/tests
   python -m pytest test_aphfw_hub.py

Without the module these tests are skipped, not failed. The MPI tests use
Pyomo's ``gurobi`` interface on the Python side, so they need ``gurobipy``,
and one of them needs 6 MPI slots.

Running
-------

Select the hub with ``--aphfw-hub`` in ``generic_cylinders`` (see
:ref:`generic_cylinders`) and add spokes as with any other hub::

   mpiexec -np 3 python -m mpi4py -m mpisppy.generic_cylinders \
       --module-name mpisppy.tests.examples.farmer --num-scens 3 \
       --solver-name gurobi_persistent --aphfw-hub --default-rho 1 \
       --aphfw-max-cycles 50 --lagrangian --xhatshuffle --rel-gap 0.001

The hub's subproblems are solved in C++ by the solver the module was built
for. ``--solver-name`` is still needed, for the spokes; it may name a
different solver. The C++ solves take only a MIP gap, a time limit and a
thread count, set by these options:

.. list-table:: Settings for the hub's C++ solves
   :widths: 30 20 50
   :header-rows: 1

   * - Option
     - Default
     - Applies to
   * - ``--iter0-mipgap``
     - the solver's
     - The Lagrangian solves, at startup and (with
       ``--aphfw-final-lagrangian``) at the end
   * - ``--iterk-mipgap``
     - :math:`10^{-4}`
     - The Frank-Wolfe linear subproblems
   * - ``--aphfw-solve-time-limit``
     - 300
     - Every C++ solve, in seconds
   * - ``--max-solver-threads``
     - 1
     - Every C++ solve

As for the other hubs, ``--iter0-mipgap`` and ``--iterk-mipgap`` exist only
when the model's ``inparser_adder`` calls ``cfg.mip_options()``, and they
also reach the spokes. ``--solver-options`` and ``--solver-options-file``
do not reach the C++ solves; the hub says so in a warning, and they still
apply to the spokes.

The hub's own options all begin with ``--aphfw-``. It also takes
``--default-rho``, and the step size ``--aph-nu`` shared with
:ref:`sec-aph`. The ones most often set:

.. list-table:: FW-APH hub options
   :widths: 30 10 60
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--default-rho``
     -
     - Proximal term multiplier, as for PH and APH
   * - ``--aph-nu``
     - 1.0
     - Step size multiplier (shared with APH)
   * - ``--aphfw-max-cycles``
     - 0
     - Maximum number of cycles (0 = no limit)
   * - ``--aphfw-fw-tmax``
     - 1
     - Maximum Frank-Wolfe iterations (linear subproblem solves) per
       scenario solve
   * - ``--aphfw-fw-tau``
     - 0.01
     - Frank-Wolfe gap at which a scenario solve stops early
   * - ``--aphfw-primal-term-thresh``
     - 0.01
     - Primal termination tolerance (on :math:`u`)
   * - ``--aphfw-dual-term-thresh``
     - 0.01
     - Dual termination tolerance (on :math:`v`)
   * - ``--aphfw-gamma-inverse``
     - 1.0
     - Primal/dual scaling (applied to the norm of :math:`v`)
   * - ``--aphfw-solve-fraction``
     - 1.0
     - Fraction of the subproblem solves needed before a projection
   * - ``--aphfw-block-sync``
     - off
     - Allow only block asynchronism
   * - ``--aphfw-final-lagrangian``
     - off
     - After the main loop, solve a Lagrangian bound in the hub and let it
       compete with the bound spokes
   * - ``--aphfw-verbosity``
     - 0
     - C++ output; 1 adds a line per subproblem solve and per cycle

``--help`` (with a ``--module-name``) lists the rest. ``--aphfw-hub``
cannot be combined with ``--integer-relax-then-enforce``: the C++ hub solves
from the MPS files and does not see changes to Pyomo variable types.

Troubleshooting
---------------

``Its C++ sources come only with a clone of the mpi-sppy repository`` / ``No aphfw_cpp sources in ...``
   mpi-sppy was installed from PyPI. Clone it from GitHub, install it from
   the clone, and build there.

``No solver found`` / ``More than one solver is set``
   Pass ``--solver`` and ``--solver-home``, or set exactly one of
   ``GUROBI_HOME``, ``CPLEX_ROOT``, ``XPRESS_HOME``.

``No include/gurobi_c++.h under ...`` (and the CPLEX and Xpress equivalents)
   ``--solver-home`` must be the solver's installation directory: for Gurobi
   the platform directory holding ``include/`` and ``lib/``, for CPLEX the
   ``CPLEX_Studio...`` directory holding ``cplex/`` and ``concert/``, for
   Xpress the directory holding ``include/xprs.h`` and ``lib/``. The message
   names what is missing.

``aphfw_cpp loads ... but mpi4py loads ...``
   The module was compiled against a different MPI than mpi4py's. Rebuild
   with ``--cxx`` set to the ``mpicxx`` of mpi4py's MPI.

``Name the solver to build for``
   A plain pip build was run without ``-C cmake.define.USE_GUROBI=ON`` (or
   ``USE_CPLEX`` / ``USE_XPRESS``). Name the solver and its home, or use the
   build command.

``pip install failed``
   The CMake or compiler error is printed above this line. The usual causes
   are a compiler without C++23 support and a solver version the build does
   not look for.

``The aphfw hub (--aphfw-hub) needs the compiled aphfw_cpp module``
   ``aphfw_cpp`` cannot be imported in the Python running the hub. Build it
   in that environment.
