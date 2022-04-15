.. _no mpi4py:

Running without MPI and mpi4py
==============================

Most of the examples and documentation assume that MPI and
mpi4py are installed. However, the examples to
solve the EF directly will work without mpi4py and it is also
possible to use PH. See ``examples.farmer.from_pysp.concrete_ampl.py`` for
one example.  It is not possible to use hubs and spokes without
mpi4py.
