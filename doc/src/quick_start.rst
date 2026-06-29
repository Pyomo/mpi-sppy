Quick Start
===========

.. _Installation:

Installation
------------

We strongly recommend installing mpi-sppy from its GitHub repository rather
than from PyPI. mpi-sppy is under active development and the PyPI release
is almost always significantly out of date; bug fixes and new features land
on GitHub first.

The repository is at https://github.com/Pyomo/mpi-sppy.

Prerequisites (all platforms)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Python 3.9 or newer
* ``git``
* A Pyomo-compatible solver (e.g., cplex, gurobi, or xpress) for most
  algorithms.
* For decomposition algorithms (PH, APH, L-shaped, …): a working MPI
  implementation and ``mpi4py``. If you only need to solve the extensive
  form directly, MPI is *not* required.


Install from GitHub on Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Install an MPI implementation. Pick the method that matches your
   environment:

   * Debian / Ubuntu:

     .. code-block:: bash

        sudo apt-get update
        sudo apt-get install -y libopenmpi-dev openmpi-bin build-essential

   * Fedora / RHEL / Rocky:

     .. code-block:: bash

        sudo dnf install openmpi openmpi-devel
        # then make mpicc / mpiexec visible in this shell:
        module load mpi/openmpi-x86_64

   * Conda environment (works on any Linux distro):

     .. code-block:: bash

        conda install -c conda-forge openmpi mpi4py

     If you used the conda path, you can skip the ``[mpi]`` extra in step 3.

2. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/Pyomo/mpi-sppy.git
      cd mpi-sppy

3. Install in editable mode with the ``mpi`` extra (quote ``".[mpi]"`` so
   shells that glob brackets, such as zsh, do not mangle it):

   .. code-block:: bash

      pip install -e ".[mpi]"

4. Install a solver of your choice (e.g. ``pip install gurobipy``; commercial
   solvers also need a license).

5. Verify the installation (see :ref:`Verify Installation` below).


Install from GitHub on macOS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Install an MPI implementation. Two reliable paths:

   * Homebrew (Intel or Apple Silicon):

     .. code-block:: bash

        brew install open-mpi

   * Conda environment (recommended on Apple Silicon to avoid wheel/build
     mismatches):

     .. code-block:: bash

        conda install -c conda-forge openmpi mpi4py

     If you used the conda path, you can skip the ``[mpi]`` extra in step 3.

2. Make sure command-line developer tools are present (one-time):

   .. code-block:: bash

      xcode-select --install

3. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/Pyomo/mpi-sppy.git
      cd mpi-sppy

4. Install in editable mode with the ``mpi`` extra (quote ``".[mpi]"`` so
   shells that glob brackets, such as zsh -- the macOS default -- do not
   mangle it):

   .. code-block:: bash

      pip install -e ".[mpi]"

5. Install a solver of your choice.

6. Verify the installation (see :ref:`Verify Installation` below).


Install from GitHub on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two installation paths are supported on Windows. Most users will have a
substantially easier time with WSL2, because the Python+MPI ecosystem is
developed and tested on Linux first. Native Windows with MS-MPI does work
but breaks more often and requires more manual setup. Detailed instructions
for both follow.

Optional: notes for Visual Studio Code users
""""""""""""""""""""""""""""""""""""""""""""

**The use of VS-Code is entirely optional.** It is intended only for users
who have already chosen to use Microsoft Visual Studio Code (VS Code) as
their editor and want it to work smoothly with the WSL2 or MS-MPI paths
below. You do not need VS Code to install or use mpi-sppy; any editor
(or no editor) is fine. If you do not use VS Code, skip this subsection
entirely and go straight to WSL2 or MS-MPI.

If you *are* using VS Code, the following one-time setup pairs well with
either installation path:

1. From the Extensions panel (``Ctrl+Shift+X``), install:

   * **Python** (publisher: Microsoft) -- language support, debugging,
     interpreter selection.
   * **Pylance** (publisher: Microsoft) -- usually installed automatically
     with Python; provides fast type checking and autocompletion.
   * **WSL** (publisher: Microsoft) -- only needed if you plan to use the
     WSL2 path below. Lets VS Code open and edit files *inside* the Ubuntu
     environment as if they were local.
   * **Jupyter** (publisher: Microsoft) -- optional, useful if you intend
     to use notebooks for analysis.

2. After you complete the WSL2 or MS-MPI steps below and the
   ``git clone`` step has produced an ``mpi-sppy`` folder, open it in
   VS Code:

   * **If you used WSL2:** from the Ubuntu shell, ``cd mpi-sppy`` and run
     ``code .`` -- VS Code will launch, install its WSL server in the
     background the first time, and open the folder. The lower-left
     status bar should read "WSL: Ubuntu" to confirm you are editing
     *inside* WSL, not on Windows.
   * **If you used native MS-MPI:** open VS Code and choose
     "File → Open Folder…", then select the cloned ``mpi-sppy`` directory.

3. Select the correct Python interpreter. Open the Command Palette with
   ``Ctrl+Shift+P``, type "Python: Select Interpreter", and press Enter.
   Pick the interpreter from the virtual environment you created in the
   instructions below (e.g. ``~/mpisppy-env/bin/python`` for WSL2 or
   ``mpisppy-env\Scripts\python.exe`` for native Windows). VS Code will
   remember this choice for the folder.

4. Open an integrated terminal with ``Ctrl+` `` (Ctrl plus backtick). On
   WSL2 this gives you the Ubuntu shell directly; on native Windows it
   gives you PowerShell. All ``mpiexec`` and ``python`` commands shown
   later in this document can be typed into this terminal.

Continue with either WSL2 or native MS-MPI below.

WSL2 (Windows Subsystem for Linux)
""""""""""""""""""""""""""""""""""

1. Install WSL2 with an Ubuntu distribution. From an *administrator*
   PowerShell:

   .. code-block:: powershell

      wsl --install -d Ubuntu

   Reboot if Windows asks you to, then launch the new "Ubuntu" entry from
   the Start menu and finish the first-time user setup.

2. From inside the Ubuntu shell, install Python, git, OpenMPI, and a build
   toolchain:

   .. code-block:: bash

      sudo apt-get update
      sudo apt-get install -y python3 python3-pip python3-venv git \
          libopenmpi-dev openmpi-bin build-essential

3. (Recommended) Create and activate a Python virtual environment so
   mpi-sppy and its dependencies do not interfere with the system Python:

   .. code-block:: bash

      python3 -m venv ~/mpisppy-env
      source ~/mpisppy-env/bin/activate

4. Clone the repository and install:

   .. code-block:: bash

      git clone https://github.com/Pyomo/mpi-sppy.git
      cd mpi-sppy
      pip install -e ".[mpi]"

5. Install your solver inside the WSL environment (e.g.
   ``pip install gurobipy``). Commercial solver licenses generally work
   cross-platform.

6. Verify the installation (see :ref:`Verify Installation` below). All
   ``mpiexec``/``python`` commands should be run from the Ubuntu shell, not
   from PowerShell.

Native Windows with MS-MPI
""""""""""""""""""""""""""

This path uses Microsoft MPI (MS-MPI), the standard MPI implementation on
Windows.

1. Install Python 3.9 or newer from https://www.python.org/downloads/ or
   via Miniconda. If using the python.org installer, check
   "Add python.exe to PATH" during install.

2. Install Git for Windows from https://git-scm.com/download/win. This
   provides ``git`` and a "Git Bash" shell (you can use either PowerShell
   or Git Bash for the steps below).

3. Install Microsoft MPI. You need *both* downloads from the Microsoft MPI
   downloads page
   (https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi):

   * ``msmpisetup.exe`` -- the MPI runtime (provides ``mpiexec.exe``)
   * ``msmpisdk.msi`` -- the MPI SDK (headers/libs needed to build
     ``mpi4py``)

   After installing both, open a *new* PowerShell window and confirm that
   ``mpiexec`` is on PATH:

   .. code-block:: powershell

      mpiexec -help

4. (Recommended) Create and activate an isolated environment and install
   ``mpi4py`` into it. Pick **one** of the following paths; do not mix
   them, because ``conda install`` puts packages in the active conda
   environment rather than a ``venv``.

   * **conda-forge (simplest).** Installs a prebuilt ``mpi4py``, so you do
     not need the MS-MPI SDK or a C++ compiler:

     .. code-block:: powershell

        conda create -n mpisppy-env python=3.12
        conda activate mpisppy-env
        conda install -c conda-forge mpi4py

   * **venv + pip.** Use this only if you have the Microsoft C++ Build
     Tools and the MS-MPI SDK installed, since pip builds ``mpi4py`` from
     source:

     .. code-block:: powershell

        py -m venv mpisppy-env
        mpisppy-env\Scripts\Activate.ps1
        pip install mpi4py

5. Clone the repository and install mpi-sppy. Note that in PowerShell the
   square brackets in ``[mpi]`` must be quoted:

   .. code-block:: powershell

      git clone https://github.com/Pyomo/mpi-sppy.git
      cd mpi-sppy
      pip install -e ".[mpi]"

6. Install your solver. The Python bindings for gurobi and cplex are
   available natively on Windows.

7. Verify the installation (see :ref:`Verify Installation` below). On
   native Windows, use MS-MPI's ``mpiexec`` form, for example:

   .. code-block:: powershell

      mpiexec -n 3 python -m mpi4py mpisppy\generic_cylinders.py ...


Install from PyPI (not recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A released version of mpi-sppy is also published on PyPI:

.. code-block:: bash

   pip install mpi-sppy[mpi]

This release typically lags the GitHub ``main`` branch by months; bug
fixes and new features may not yet be available. We recommend the GitHub
install above for any active research use.


.. _Verify Installation:

Verify Installation
-------------------

The following three checks confirm that mpi-sppy, your solver, and (if
you installed it) MPI are all working. The commands below are
shell-neutral: they work in bash, zsh, the WSL2 Ubuntu shell, and
Windows PowerShell. On native Windows, if ``python`` is not on PATH but
the Python launcher is, substitute ``py`` for ``python``. Start from the
top of the cloned ``mpi-sppy`` repository; step 1 changes into
``examples/farmer`` and the remaining commands are run from there.

1. **mpi-sppy is importable and the CLI works.** This confirms the
   editable install succeeded and Python can import the package.

   .. code-block:: text

      cd examples/farmer
      python -m mpisppy.generic_cylinders --module-name farmer --help

   You should see a long help message listing all available
   command-line options. If you see ``ModuleNotFoundError: No module
   named 'mpisppy'``, the install did not take effect in this
   environment (most often a virtual-environment issue).

2. **Your solver works.** Solve the farmer extensive form directly
   with three scenarios. Substitute the solver you installed
   (``gurobi``, ``cplex``, ``xpress``, …):

   .. code-block:: text

      python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 --EF --EF-solver-name gurobi

   You should see the solver print progress and the script print an
   optimal objective value. This check does *not* use MPI.

3. **MPI works** (only if you installed MPI and ``mpi4py``). Still in
   ``examples/farmer``, run the bundled one-sided MPI test (its path is
   given relative to ``examples/farmer``):

   .. code-block:: text

      mpiexec -n 2 python -m mpi4py ../../mpi_one_sided_test.py

   If you see no error messages, your MPI installation should be
   suitable. Then confirm the full hub-and-spoke flow with a short PH
   run on farmer:

   .. code-block:: text

      mpiexec -n 3 python -m mpi4py -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 --solver-name gurobi --max-iterations 5 --default-rho 1 --lagrangian --xhatshuffle

   You should see iteration output and the run should terminate
   normally.

If you only intend to solve the extensive form directly, you can skip
step 3 -- MPI is not required for ``--EF``. For additional MPI install
guidance and HPC-specific tips, see :ref:`Install mpi4py`.


Running the Farmer Example
---------------------------

**Recommended first run: let mpi-sppy configure itself.** Add
``--out-of-the-box`` and the driver introspects the environment and the model and
picks a sensible configuration (solver, EF vs. decomposition, spokes, bundling),
prints the equivalent explicit command line, and runs it:

.. code-block:: bash

   python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 \
       --out-of-the-box

Any option you set explicitly always wins, so this is a good starting point you
can refine. (For a small, fast-solving model like farmer, OOTB will sensibly
choose the extensive form; it decomposes for larger or harder problems.) See
:ref:`out_of_the_box` for the full description -- effort tiers,
``--inspect-only``, policy files, and the validation and calibration tools.

The explicit forms below show what such a run is equivalent to.

**Solve the EF** (does not use MPI):

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


What You Need to Provide For Your Problem
-----------------------------------------

If your model is written in Pyomo, you create a Python module with the
following functions:

- ``scenario_creator`` -- builds a Pyomo model for one scenario (see :ref:`scenario_creator`)
- ``scenario_names_creator`` -- returns the list of scenario names (see :ref:`helper_functions`)
- ``kw_creator`` -- returns keyword arguments for the scenario creator (see :ref:`helper_functions`)
- ``inparser_adder`` -- adds problem-specific command-line arguments (see :ref:`helper_functions`)
- ``scenario_denouement`` -- called at termination (can be ``None``; see :ref:`helper_functions`)

Once you have these functions, you can use ``generic_cylinders.py``
(see :ref:`generic_cylinders`) to solve your problem using the EF or
the hub-and-spoke system. See the ``farmer`` directory in ``examples``
for a complete working example (``farmer.py`` and ``farmer_generic.bash``).

For models written in an algebraic modeling language other than Pyomo
(e.g., AMPL or GAMS), see :doc:`agnostic`. For models supplied as
SMPS-format files, see :doc:`smps`.
