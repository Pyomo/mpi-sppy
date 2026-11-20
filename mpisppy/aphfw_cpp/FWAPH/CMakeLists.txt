###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# aph_core_<backend>: the FW-APH controller library. Everything the
# aphfw_cpp module needs lives under aphfw_cpp/; the standalone
# executables built on this library are in aph-fw's src/FWAPH.

set(APH_SRCS
    FWAPH.cpp
    groupController.cpp
    ProblemWithSolvers.cpp
    scenario.cpp
    spControllerBase.cpp
    appRun.cpp
)

set(APH_COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../config    
)

# --------------------------
# GUROBI variant
# --------------------------
if(USE_GUROBI)
    add_library(aph_core_gurobi STATIC ${APH_SRCS})
    target_include_directories(aph_core_gurobi PUBLIC ${APH_COMMON_INCLUDES})
    target_link_libraries(aph_core_gurobi PUBLIC aph_utils solver_gurobi)
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(aph_core_gurobi PUBLIC MPI::MPI_CXX)
    endif()
    target_compile_definitions(aph_core_gurobi PUBLIC USE_GUROBI)
    if(NODEBUGPRINT)
        target_compile_definitions(aph_core_gurobi PUBLIC NODEBUGPRINT)
    endif()
endif()

# --------------------------
# CPLEX variant
# --------------------------
if(USE_CPLEX)
    add_library(aph_core_cplex STATIC ${APH_SRCS})
    target_include_directories(aph_core_cplex PUBLIC ${APH_COMMON_INCLUDES})
    target_link_libraries(aph_core_cplex PUBLIC aph_utils solver_cplex)
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(aph_core_cplex PUBLIC MPI::MPI_CXX)
    endif()
    target_compile_definitions(aph_core_cplex PUBLIC USE_CPLEX)
    if(NODEBUGPRINT)
        target_compile_definitions(aph_core_cplex PUBLIC NODEBUGPRINT)
    endif()
endif()

# --------------------------
# XPRESS variant
# --------------------------
if(USE_XPRESS)
    add_library(aph_core_xpress STATIC ${APH_SRCS})
    target_include_directories(aph_core_xpress PUBLIC ${APH_COMMON_INCLUDES})
    target_link_libraries(aph_core_xpress PUBLIC aph_utils solver_xpress)
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(aph_core_xpress PUBLIC MPI::MPI_CXX)
    endif()
    target_compile_definitions(aph_core_xpress PUBLIC USE_XPRESS)
    if(NODEBUGPRINT)
        target_compile_definitions(aph_core_xpress PUBLIC NODEBUGPRINT)
    endif()
endif()

# --------------------------
# HiGHS variant
# --------------------------
if(USE_HIGHS)
    add_library(aph_core_highs STATIC ${APH_SRCS})
    target_include_directories(aph_core_highs PUBLIC ${APH_COMMON_INCLUDES})
    target_link_libraries(aph_core_highs PUBLIC aph_utils solver_highs)
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(aph_core_highs PUBLIC MPI::MPI_CXX)
    endif()
    target_compile_definitions(aph_core_highs PUBLIC USE_HIGHS)
    if(NODEBUGPRINT)
        target_compile_definitions(aph_core_highs PUBLIC NODEBUGPRINT)
    endif()
endif()
