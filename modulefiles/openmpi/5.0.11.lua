-- Private Open MPI 5.0.11 installation for this mpi-sppy checkout.
--
-- Usage:
--   module use /g/g11/colekend/vast/dane/mpi-sppy/modulefiles
--   module load openmpi/5.0.11
--
-- Load this before creating or activating the matching Python/mpi4py
-- environment.  Do not combine it with a system Open MPI module.

help([[
Loads the private Open MPI 5.0.11 installation built for this mpi-sppy
checkout.  Use its mpicc wrapper when building mpi4py, then activate only the
virtual environment built against this exact MPI installation.
]])

whatis("Name: Open MPI")
whatis("Version: 5.0.11")
whatis("Category: MPI")
whatis("Description: Private Open MPI 5.0.11 installation for mpi-sppy")

local root = "/g/g11/colekend/vast/dane/mpi-sppy/.local/openmpi-5.0.11"

if not isDir(root) then
    LmodError("Private Open MPI 5.0.11 is not installed at " .. root ..
              ". Build it first with build_openmpi_5.0.11.sh.")
end

family("mpi")

prepend_path("PATH",            pathJoin(root, "bin"))
prepend_path("LD_LIBRARY_PATH", pathJoin(root, "lib"))
prepend_path("LIBRARY_PATH",    pathJoin(root, "lib"))
prepend_path("CPATH",           pathJoin(root, "include"))
prepend_path("PKG_CONFIG_PATH", pathJoin(root, "lib", "pkgconfig"))
prepend_path("CMAKE_PREFIX_PATH", root)
prepend_path("MANPATH",         pathJoin(root, "share", "man"))

setenv("MPI_HOME", root)
setenv("OMPI_HOME", root)
setenv("OMPI_ROOT", root)
setenv("OPAL_PREFIX", root)
