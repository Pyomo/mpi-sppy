.. _Install mpi4py:

Install mpi4py
==============

A recent version of MPI and a compatible version of ``mpi4py`` are needed for parallel operation and ``mpi4py`` is assumed to be available for import by most of
the code-base (an exception is :ref:`sputils.create_EF`).

Here are two methods that seem to work well for installation, at least when considering non-HPC platforms.

#. Install OpenMPI and mpi4py using conda.

   * ``conda install openmpi; conda install mpi4py``  (in that order)
  
#. If you already have an existing version of MPI, it may be better compile mpi4py against it. This can be done by installing mpi4py through pip.

   * ``pip install mpi4py``

#. It is also possible to automate mpi4py installation through pip  as an optional dependency when installing mpi-sppy from pip or from source by adding the `[mpi]` extras flag.

   * ``pip install -e .[mpi]`` (after cloning and moving to the repo root directory)

Native MPI progress worker
^^^^^^^^^^^^^^^^^^^^^^^^^^

On Linux and macOS, an MPI-enabled source build of mpi-sppy compiles the native
progress worker when ``mpi4py`` and a working MPI compiler are available in the
build environment. A single-process compiler probe records the MPI library
version, communicator size, and source and tag constants. It uses the native
extension's configured compile and link settings, and each native build
recompiles the extension. If that probe fails, or on other platforms, the build
uses the Python worker. An explicitly selected but missing ``CC`` is an error.
Make an sdist from the checkout, then install it
in an MPI virtualenv after installing ``mpi4py``::

   python -m pip install build
   python -m build --sdist
   python -m pip install "setuptools>=68" wheel mpi4py
   CC=mpicc python -m pip install --no-build-isolation --no-cache-dir dist/mpi_sppy-*.tar.gz

Select the ``mpicc`` that matches the MPI implementation used by ``mpi4py``.
Install the sdist separately in each virtualenv when using multiple MPI
implementations. Each pip build then has a fresh source directory and produces
an extension linked to that virtualenv's MPI stack. An editable install also
works from a shared checkout after ``mpi4py`` and the build tools are installed::

   CC=mpicc python -m pip install --no-build-isolation -e .

Editable extensions are installed into each virtualenv rather than copied into
the checkout, so virtualenvs using different MPI implementations do not share
the compiled worker. Reinstall the editable package after changing its C code
or the virtualenv's MPI installation.

The ``[mpi]`` extra alone does not enable the native build: pip builds
mpi-sppy before installing its runtime extras, and an isolated build cannot
see ``mpi4py`` from the target environment. A build without ``mpi4py`` remains
pure Python. The C sources are included in the source distribution, so it can
also be rebuilt in an MPI-enabled environment.

``MPISPPY_MPI_PROGRESS_THREAD=1`` activates the progress thread and requests
``MPI_THREAD_MULTIPLE``. If the installed package has the native worker, it is
selected after comparing its build signature with values reported by mpi4py.
A mismatch raises an error before the extension uses mpi4py's communicator.
This check detects common mismatches but cannot prove that two separately
loaded copies of an MPI library share runtime state. If no native worker is
installed, the Python worker runs. The Python worker calls mpi4py from a
thread and cannot poll while solver code holds Python's GIL; the native worker
polls without holding it. Set ``MPISPPY_MPI_PROGRESS_INTERVAL_US`` to a
positive, finite number to change the maximum polling interval from its
default of 500 microseconds.

To check the MPI runtime without importing mpi-sppy by default, give this
command (once mpi-sppy is installed it works from any directory):

``mpirun -n 2 python -m mpi4py -m mpi_one_sided_test``

With ``MPISPPY_MPI_PROGRESS_THREAD`` unset or set to ``0``, this checks the MPI
runtime's passive-target RMA progress. Set it to ``1`` to check the same operation with
mpi-sppy's progress worker active while the target is idle. The worker can
make the test pass even if the native runtime check fails.

If you don't see any error messages, you might have an MPI
installation that will work well. Note that even if there is
an error message, mpi-sppy may still execute and return correct
results. 

no MPI?
^^^^^^^

If you can't get MPI and mpi4py, some functionality is still
available. In particular, it is possible to solve the EF
directly. (See :ref:`no mpi4py` for a little more
information.)  However, most of the examples and most of the
documentation assumes you have mpi4py installed and running correctly.


AN IMPORTANT NOTE FOR MPICH USERS ON HPC PLATFORMS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
At least on some US Department of Energy (e.g., at Lawrence Livermore National Laboratory) compute clusters, users of mpi-sppy that are using an MPICH implementation of MPI may need to set the following in order for both (1) proper execution of the one-sided test referenced above and (2) rapid results when running any of the algorithms shipped with mpi-sppy:

export MPICH_ASYNC_PROGRESS=1

Without this setting, we have observed run-times increase by a factor of between 2 and 4, due to non-blocking point-to-point calls apparently being treated as blocking.

Further, without this setting and in situations with a large number of ranks (e.g., >> 10), we have observed mpi-sppy stalling once scenario instances are created.


Other pip extras
^^^^^^^^^^^^^^^^

In addition to ``[mpi]``, mpi-sppy declares other ``pip`` extras that can
be combined with it (with commas, e.g. ``pip install -e ".[mpi,dev]"``):

* ``extras`` -- every pip-installable optional feature of the library
  itself: ``scipy`` (confidence intervals, quadratic prox approximation,
  kkt utilities), ``pandas`` (rho CSV utilities, W tracking),
  ``matplotlib`` (plotting), ``dill`` (pickled scenario bundles), and
  ``mip`` (MPS/LP/SMPS file input).
* ``test`` -- everything in ``extras`` plus the tools the test suite and
  the examples need beyond a solver (``pytest``, ``coverage``,
  ``addheader``, and ``gridx-egret`` for the unit commitment examples).
* ``doc`` -- Sphinx and the theme used to build the documentation.
* ``dev`` -- the one-stop shop for contributors: everything in ``test``
  and ``doc`` plus ``ruff`` (linting).

Solvers (e.g. cplex, gurobipy, xpress) are deliberately in no extra:
which one to install is a licensing decision, so install your solver of
choice separately.
