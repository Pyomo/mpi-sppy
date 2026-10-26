-- Local modulefile for MVAPICH2 4.1 built with GCC 13.3.1.
--
-- Usage from the repository root:
--   module use "$PWD/modulefiles"
--   module load mvapich2/4.1

local mpi_root = "/collab/usr/global/tools/mpi/toss_4_x86_64_ib/mvapich2-4.1-gcc-13.3.1"

whatis("Name: MVAPICH2")
whatis("Version: 4.1")
whatis("Category: MPI")
whatis("Description: Local MVAPICH2 4.1 module built with GCC 13.3.1")

help([[\
Loads MVAPICH2 4.1 built with GCC 13.3.1 and configures the MPICH-compatible
compiler wrapper used to build mpi4py.

For the repository virtual environment:
  module use "$PWD/modulefiles"
  module load mvapich2/4.1
]])

-- Match the toolchain used by the MVAPICH2 installation and ensure that only
-- one MPI implementation is active at a time.
load("python/3.13.2")
load("gcc/13.3.1-magic")
family("mpi")

setenv("MVAPICH2_ROOT", mpi_root)
setenv("MPI_HOME", mpi_root)
prepend_path("PATH", pathJoin(mpi_root, "bin"))
prepend_path("LD_LIBRARY_PATH", pathJoin(mpi_root, "lib"))
setenv("CC", "mpicc")
setenv("CXX", "mpicxx")
setenv("FC", "mpifort")
