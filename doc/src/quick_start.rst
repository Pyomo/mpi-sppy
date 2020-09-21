Quick Start
===========

Verify installation
-------------------

Getting started depends on how you intend to use ``mpi-sspy`` but
verifying installation is a common task. If you installed mpi-sspy from
github, you can verify that you installed it and a solver by starting a
terminal. Then cd to the `mpi-sspy` directory and issue the following
terminal commands:

::
   cd mpisppy
   cd examples
   cd farmer
   python farmer_ef 1 3 solvername

but replace `solvername` with the name of the solver you have installed, e.g., if you have installed glpk, use

::
   
   python farmer_ef 1 3 glpk

If you intend to use any parallel features, you should verify that you have
a *proper* installation of MPI and `mpi4py`; see the section `Install mpi4py`_.


PySP Users
----------

If you are already using PySP for a two-stage model, getting started
with `mpi-sppy` is straightforward; however, unlike in PySP, you will
be required to create a Python program. The basic vehicle is
``mpisppy.utils.pysp_model.PySPModel`` but the exact steps depend on
how you represented your model in PySP.

Here are the general steps:

# Construct a PySPModel object giving its constructor information about your PySP model.

# Create an options dictionary.

# Create a PH or EF `mpi-sppy` object.

# Call its main function.

These steps alone will not result in use of the hub-spoke features of `mpi-sppy`, but they will
get your PySP model running in `mpi-sppy`. See ``mpisspy.examples.farmer.farmer_pysp`` for an example; see
`PySP conversion`_ for more details.


Pyomo Users who want to add stochastics
---------------------------------------

The first thing is to code a scenario creation function. See `scenario_creator`_ for more information. Once
you have the function, you can mimic the code in ``mpisspy.examples.farmer.farmer_ef`` to test your function
to solve the extensive form directly.


Researchers who want to compare with mpi-sppy
---------------------------------------------

The quickest thing to do is to run one of the canned examples that
comes with `mpi-sppy`. They are in subdirectories of
`mpisppy.examples` and sample commands can be obtained by looking at
the code in `mpisppy.examples.runall`. There is a table in the
mpi-sppy paper that gives references for all of the examples.
