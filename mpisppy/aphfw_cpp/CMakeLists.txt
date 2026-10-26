# Standalone build of the aphfw_cpp Python module (FW-APH hub for mpi-sppy).
#
# This directory is shared with mpi-sppy (as mpisppy/aphfw_cpp, via git
# subtree), so it must build on its own: just the module and the static
# libraries it links, with no standalone executables and no tests. In aph-fw
# the top-level CMakeLists.txt builds these same subdirectories alongside
# everything else.
#
#   cmake -S aphfw_cpp -B build -DUSE_GUROBI=ON -DGUROBI_HOME=... \
#         -DCMAKE_CXX_COMPILER=mpic++ -Dpybind11_DIR=...
#   cmake --build build --parallel
cmake_minimum_required(VERSION 3.16)
project(aphfw_cpp LANGUAGES CXX)

# AphfwSettings.cmake turns USE_GUROBI on by default, which suits aph-fw's own
# build but would make a pip build of this directory quietly assume Gurobi.
# Here the solver has to be named: at least one USE_* given, and any not given
# is turned off, so -DUSE_CPLEX=ON alone is enough.
set(_named_solvers "")
foreach(_solver GUROBI CPLEX XPRESS)
    if(DEFINED CACHE{USE_${_solver}})
        if(USE_${_solver})
            list(APPEND _named_solvers ${_solver})
        endif()
    else()
        set(USE_${_solver} OFF CACHE BOOL "Build the ${_solver}-backed module")
    endif()
endforeach()
if(NOT _named_solvers)
    message(FATAL_ERROR
        "Name the solver to build for: one of -DUSE_GUROBI=ON, -DUSE_CPLEX=ON "
        "or -DUSE_XPRESS=ON (with pip: -C cmake.define.USE_GUROBI=ON, etc.), "
        "together with its home (GUROBI_HOME, CPLEX_ROOT or XPRESS_HOME).  "
        "Or let python -m mpisppy.aphfw_cpp.build choose and check it.")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AphfwSettings.cmake)

add_subdirectory(utils)
add_subdirectory(solver_interfaces)
add_subdirectory(FWAPH)
add_subdirectory(python)
