mpi-sppy
========

Optimization under uncertainty for `Pyomo <https://pyomo.org>`_ models.

`Documentation is available at readthedocs <https://mpi-sppy.readthedocs.io/en/latest/>`_ and
a there is a `paper <https://link.springer.com/article/10.1007/s12532-023-00247-3>`_

Status for internal tests
^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: https://github.com/Pyomo/mpi-sppy/workflows/pyotracker/badge.svg
   :target: https://github.com/Pyomo/mpi-sppy/actions/workflows/pyotracker.yml


MPI
^^^

A recent version of MPI and a compatible version of mpi4py are needed.

Here are two methods that seem to work well for installation, at least when considering non-HPC platforms.

#. Install OpenMPI and mpi4py using conda.

   * ``conda install openmpi; conda install mpi4py``  (in that order)
  
#. If you already have an existing version of MPI, it may be better compile mpi4py against it. This can be done by installing mpi4py though pip.

   * ``pip install mpi4py``

To test
your installation, cd to the directory where you installed mpi-sppy
(it is called ``mpi-sppy``) and then give this command.

``mpirun -n 2 python -m mpi4py mpi_one_sided_test.py``

If you don't see any error messages, you might have an MPI
installation that will work well. Note that even if there is
an error message, mpi-sppy may still execute and return correct
results. Per the comment below, the run-times may just be 
unnecessarily inflated.

Installing mpi-sppy
^^^^^^^^^^^^^^^^^^^

It is possible to pip install mpi-sppy; however, most users are better off
getting the software from github because it is under active development.

Citing mpi-sppy
^^^^^^^^^^^^^^^
If you find mpi-sppy useful in your work, we kindly request that you cite the following `paper <https://link.springer.com/article/10.1007/s12532-023-00247-3>`_:

::

   @article{mpi-sppy,
     title={A Parallel Hub-and-Spoke System for Large-Scale Scenario-Based Optimization Under Uncertainty},
     author={Bernard Knueven  and David Mildebrath and Christopher Muir  and John D Siirola  and Jean-Paul Watson  and David L Woodruff},
     journal = {Math. Prog. Comp.},
     volume = {15}, 
     pages = {591-–619},
     year={2023}
   }



AN IMPORTANT NOTE FOR MPICH USERS ON HPC PLATFORMS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At least on some US Department of Energy (e.g., at Lawrence Livermore
National Laboratory) compute clusters, users of mpi-sppy that are
using an MPICH implementation of MPI may need to set the following in
order for both (1) proper execution of the one-sided test referenced
above and (2) rapid results when running any of the algorithms shipped
with mpi-sppy:

export MPICH_ASYNC_PROGRESS=1

Without this setting, we have observed run-times increase by a factor
of between 2 and 4, due to non-blocking point-to-point calls
apparently being treated as blocking.

Further, without this setting and in situations with a large number of
ranks (e.g., >> 10), we have observed mpi-sppy stalling once scenario
instances are created.

2022 NOTICE
^^^^^^^^^^^

There was a disruptive change on August 11, 2022 concerning how
options are accessed. See the file ``disruptions.txt`` for more
information. If you are a new user, this will not affect you,
regardless of how you install. If you are an
existing user, you should consider the disruption before updating to
the latest mpi-sppy. The documentation on readthedocs
probably refers to the newest version.

