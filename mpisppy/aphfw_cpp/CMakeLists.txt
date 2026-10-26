# Standalone build of the aphfw_cpp Python module (FW-APH hub for mpi-sppy).
#
# By default it builds just the module and the static libraries it links,
# which is what pip builds. With -DAPHFW_BUILD_TESTS=ON it also builds the
# standalone driver main_aph_<solver> and the C++ tests in tests/, which ctest
# then runs:
#
#   cmake -S mpisppy/aphfw_cpp -B build -DUSE_GUROBI=ON -DGUROBI_HOME=... \
#         -DCMAKE_CXX_COMPILER=mpicxx -Dpybind11_DIR=... -DAPHFW_BUILD_TESTS=ON
#   cmake --build build --parallel
#   ctest --test-dir build --output-on-failure
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

option(APHFW_BUILD_TESTS "Build the standalone driver and the C++ tests" OFF)
if(APHFW_BUILD_TESTS)
    foreach(_solver GUROBI CPLEX XPRESS)
        if(USE_${_solver})
            string(TOLOWER ${_solver} _lower)
            set(APHFW_CORE_TARGET aph_core_${_lower})
            set(APHFW_MAIN_TARGET main_aph_${_lower})
            add_executable(${APHFW_MAIN_TARGET} FWAPH/FWAPH_main.cpp)
            target_compile_definitions(${APHFW_MAIN_TARGET} PRIVATE USE_${_solver})
            if(NODEBUGPRINT)
                target_compile_definitions(${APHFW_MAIN_TARGET} PRIVATE NODEBUGPRINT)
            endif()
            target_link_libraries(${APHFW_MAIN_TARGET} PRIVATE
                ${APHFW_CORE_TARGET} solver_${_lower})
        endif()
    endforeach()
    if(TARGET GUROBI::cxx AND TARGET GUROBI::core)
        target_link_options(${APHFW_MAIN_TARGET} PRIVATE -Wl,--no-as-needed)
        target_link_libraries(${APHFW_MAIN_TARGET} PRIVATE GUROBI::cxx GUROBI::core)
    endif()
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(${APHFW_MAIN_TARGET} PRIVATE MPI::MPI_CXX)
    endif()
    enable_testing()
    add_subdirectory(tests)
endif()
