mpi-sppy
========

Optimization under uncertainty for Pyomo models.

MPI
^^^

A recent version of MPI and a compatible version of mpi4py are needed.

Here are two methods that seem to work well for installation.

#. Install OpenMPI and mpi4py using conda.

   * ``conda install openmpi; conda install mpi4py``  (in that order)
  
#. If you already have an existing version of MPI, it may be better compile mpi4py against it. This can be done by installing mpi4py though pip.

   * ``pip install mpi4py``

To test
your installation, cd to the directory where you installed mpi-sppy
(it is called ``mpi-sppy``) and then give this command.

``mpirun -n 2 python -m mpi4py mpi_one_sided_test.py``

If you don't see any error messages, you might have an MPI
installation that will work well.
