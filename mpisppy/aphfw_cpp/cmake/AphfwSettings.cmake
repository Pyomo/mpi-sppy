###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Settings shared by aph-fw's top-level CMakeLists.txt and the standalone
# aphfw_cpp/CMakeLists.txt: language level, the solver backend switch,
# MPI, and where each solver is installed. Include it after project().

# -------------------------
# Global compile settings
# -------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------
# User options
# ---------------------------------
option(USE_GUROBI    "Build GUROBI-enabled binaries" ON)
option(USE_CPLEX     "Build CPLEX-enabled binaries"  OFF)
option(USE_XPRESS    "Build XPRESS-enabled binaries" OFF)
option(NODEBUGPRINT  "Disable debug prints"           OFF)

# Don't allow multiple solvers at once
set(_SOLVER_COUNT 0)
if(USE_GUROBI)
    math(EXPR _SOLVER_COUNT "${_SOLVER_COUNT} + 1")
endif()
if(USE_CPLEX)
    math(EXPR _SOLVER_COUNT "${_SOLVER_COUNT} + 1")
endif()
if(USE_XPRESS)
    math(EXPR _SOLVER_COUNT "${_SOLVER_COUNT} + 1")
endif()
if(_SOLVER_COUNT GREATER 1)
    message(FATAL_ERROR
        "Multiple solver backends are ON.\n"
        "Please enable at most one of USE_GUROBI, USE_CPLEX, USE_XPRESS.")
endif()

# ---------------------------------
# MPI (required)
# ---------------------------------
find_package(MPI REQUIRED CXX)

# ---------------------------------
# External solver roots
# ---------------------------------
# GUROBI_HOME: expected by aphfw_cpp/solver_interfaces/CMakeLists.txt
set(GUROBI_HOME "" CACHE PATH
    "Path to GUROBI installation (root containing include/ and lib/)")

# Import from environment if the cache value is empty
if(GUROBI_HOME STREQUAL "" AND DEFINED ENV{GUROBI_HOME})
    set(GUROBI_HOME "$ENV{GUROBI_HOME}" CACHE PATH
        "Path to GUROBI installation (root containing include/ and lib/)" FORCE)
endif()

message(STATUS "GUROBI_HOME = '${GUROBI_HOME}'")

# ---------------------------------
# CPLEX root from old variable name
# ---------------------------------
set(CPLEX_ROOT "" CACHE PATH
    "Path to CPLEX_Studio* directory (root with cplex/ and concert/)")
set(CPLEX_ROOT_DIR "${CPLEX_ROOT}" CACHE PATH
    "Alias: root of CPLEX installation (used by solver_interfaces)" FORCE)

# ---------------------------------
# XPRESS root
# ---------------------------------
set(XPRESS_HOME "" CACHE PATH
    "Path to Xpress installation (root containing include/ and lib/)")

# Import from environment if the cache value is empty
if(XPRESS_HOME STREQUAL "" AND DEFINED ENV{XPRESSDIR})
    set(XPRESS_HOME "$ENV{XPRESSDIR}" CACHE PATH
        "Path to Xpress installation (from XPRESSDIR env)" FORCE)
endif()

if(USE_XPRESS)
    message(STATUS "XPRESS_HOME = '${XPRESS_HOME}'")
endif()
