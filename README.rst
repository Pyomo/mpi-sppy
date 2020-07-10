mpi-sppy
========

Optimization under uncertainty for Pyomo models.

MPI
^^^

Install MPI and then install mpi4py using pip (not Anaconda). To test
your installation, cd to the directory where you installed mpi-sppy
(it is called ``mpi-sppy``) and then give this command.

::
   mpirun -n 2 python -m mpi4py mpi_one_sided_test.py

If you don't see any error messages, you might have an MPI
installation that will work well.
