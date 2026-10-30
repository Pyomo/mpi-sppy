###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# ================================
# aphfw_cpp/solver_interfaces/CMakeLists.txt
# ================================

# -----------------------
# Common sources (NO main())
# -----------------------
set(SOLVER_COMMON_SRCS
    solver_interface.cpp
    params.cpp
    nl_coeffs.cpp
)

add_library(solver_common STATIC ${SOLVER_COMMON_SRCS})

target_include_directories(solver_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}           # aphfw_cpp/solver_interfaces
        ${CMAKE_CURRENT_SOURCE_DIR}/..        # aphfw_cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils  # aphfw_cpp/utils for timer.hpp, etc.
)

target_link_libraries(solver_common PUBLIC aph_utils)

if(NODEBUGPRINT)
    target_compile_definitions(solver_common PUBLIC NODEBUGPRINT)
endif()

if(TARGET MPI::MPI_CXX)
    target_link_libraries(solver_common PUBLIC MPI::MPI_CXX)
endif()

# Propagate solver macro to common code
if(USE_GUROBI)
    target_compile_definitions(solver_common PUBLIC USE_GUROBI)
elseif(USE_CPLEX)
    target_compile_definitions(solver_common PUBLIC USE_CPLEX)
elseif(USE_XPRESS)
    target_compile_definitions(solver_common PUBLIC USE_XPRESS)
elseif(USE_HIGHS)
    target_compile_definitions(solver_common PUBLIC USE_HIGHS)
endif()

# ============================================================
# GUROBI variant: solver_gurobi
# ============================================================
# If GUROBI_HOME not set as a CMake var, try environment
if(NOT DEFINED GUROBI_HOME AND DEFINED ENV{GUROBI_HOME})
    set(GUROBI_HOME "$ENV{GUROBI_HOME}" CACHE PATH "Gurobi root directory")
endif()

if(USE_GUROBI AND NOT DEFINED GUROBI_HOME)
    message(FATAL_ERROR "USE_GUROBI=ON but GUROBI_HOME not set.")
endif()

if(USE_GUROBI)
    if(NOT GUROBI_HOME)
        message(FATAL_ERROR "USE_GUROBI=ON but GUROBI_HOME not set.")
    endif()

    # Restrict search to GUROBI_HOME to avoid picking system libs
    set(_GRB_DIRS
        "${GUROBI_HOME}/linux64/lib"
        "${GUROBI_HOME}/lib"
        "${GUROBI_HOME}/lib64"
    )

    # Supported core versions: 13.0 first (current licensed line), then 12.0.
    set(_GRB_CORE_CANDIDATES libgurobi130.so libgurobi120.so)
    set(_GRB_CORE "")
    foreach(_d IN LISTS _GRB_DIRS)
        foreach(_name IN LISTS _GRB_CORE_CANDIDATES)
            if(EXISTS "${_d}/${_name}")
                set(_GRB_CORE "${_d}/${_name}")
                break()
            endif()
        endforeach()
        if(_GRB_CORE)
            break()
        endif()
    endforeach()
    if(NOT _GRB_CORE)
        message(FATAL_ERROR "Could not find any of ${_GRB_CORE_CANDIDATES} under ${GUROBI_HOME} (looked in: ${_GRB_DIRS})")
    endif()

    # Prefer shared C++ wrapper; fall back to static
    set(_GRB_CXX "")
    foreach(_d IN LISTS _GRB_DIRS)
        if(EXISTS "${_d}/libgurobi_c++.so")
            set(_GRB_CXX "${_d}/libgurobi_c++.so")
            break()
        endif()
    endforeach()
    if(NOT _GRB_CXX)
        foreach(_d IN LISTS _GRB_DIRS)
            if(EXISTS "${_d}/libgurobi_c++.a")
                set(_GRB_CXX "${_d}/libgurobi_c++.a")
                break()
            elseif(EXISTS "${_d}/libgurobi_c++11.a")
                set(_GRB_CXX "${_d}/libgurobi_c++11.a")
                break()
            endif()
        endforeach()
    endif()
    if(NOT _GRB_CXX)
        message(FATAL_ERROR "Could not find libgurobi_c++(.so|.a) under ${GUROBI_HOME} (looked in: ${_GRB_DIRS})")
    endif()

    message(STATUS "Gurobi core  : ${_GRB_CORE}")
    message(STATUS "Gurobi C++   : ${_GRB_CXX}")
    message(STATUS "Gurobi inc   : ${GUROBI_HOME}/include")

    # Imported targets using absolute, version-pinned files
    add_library(GUROBI::cxx UNKNOWN IMPORTED)
    set_target_properties(GUROBI::cxx PROPERTIES IMPORTED_LOCATION "${_GRB_CXX}")

    add_library(GUROBI::core UNKNOWN IMPORTED)
    set_target_properties(GUROBI::core PROPERTIES IMPORTED_LOCATION "${_GRB_CORE}")

    # GUROBI sources in subdirectory
    set(GUROBI_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/gurobi/gurobi_interface.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gurobi/nl_gurobi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gurobi/grad_gurobi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gurobi/gurobi_params.cpp
    )

    add_library(solver_gurobi STATIC ${GUROBI_SRCS})

    target_include_directories(solver_gurobi
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/gurobi
            ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils
            ${GUROBI_HOME}/include
    )

    # Keep core even with --as-needed
    target_link_options(solver_gurobi PUBLIC -Wl,--no-as-needed)

    target_link_libraries(solver_gurobi
        PUBLIC
            solver_common
            GUROBI::cxx
            GUROBI::core
    )

    if(TARGET MPI::MPI_CXX)
        target_link_libraries(solver_gurobi PUBLIC MPI::MPI_CXX)
    endif()

    target_compile_definitions(solver_gurobi PUBLIC USE_GUROBI)
    if(NODEBUGPRINT)
        target_compile_definitions(solver_gurobi PUBLIC NODEBUGPRINT)
    endif()
endif()

# ============================================================
# CPLEX variant: solver_cplex
# ============================================================
if(USE_CPLEX)
    if(NOT CPLEX_ROOT_DIR)
        message(FATAL_ERROR "USE_CPLEX=ON but CPLEX_ROOT_DIR not set. Point it at your CPLEX_Studio* root.")
    endif()

    # Headers are split between cplex/include and concert/include
    set(CPLEX_CPLEX_INCLUDE_DIR   "${CPLEX_ROOT_DIR}/cplex/include")
    set(CPLEX_CONCERT_INCLUDE_DIR "${CPLEX_ROOT_DIR}/concert/include")

    # Common library locations for both cplex and concert parts
    set(CPLEX_LIB_DIR_CANDIDATES
        "${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic"
        "${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_mt"
        "${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux"
        "${CPLEX_ROOT_DIR}/cplex/lib"
        "${CPLEX_ROOT_DIR}/concert/lib/x86-64_linux/static_pic"
        "${CPLEX_ROOT_DIR}/concert/lib/x86-64_linux/static_mt"
        "${CPLEX_ROOT_DIR}/concert/lib/x86-64_linux"
        "${CPLEX_ROOT_DIR}/concert/lib"
    )

    # Required libraries: ilocplex, cplex, concert
    find_library(ILOCPLEX_LIB NAMES ilocplex iloCPLEX HINTS ${CPLEX_LIB_DIR_CANDIDATES} REQUIRED)
    find_library(CPLEX_LIB    NAMES cplex                 HINTS ${CPLEX_LIB_DIR_CANDIDATES} REQUIRED)
    find_library(CONCERT_LIB  NAMES concert               HINTS ${CPLEX_LIB_DIR_CANDIDATES} REQUIRED)

    # Optional system libs
    find_package(Threads REQUIRED)       # Threads::Threads
    # CMAKE_DL_LIBS is a platform variable with the proper 'dl' on Unix
    set(_DL_LIB ${CMAKE_DL_LIBS})

    # CPLEX sources
    set(CPLEX_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/cplex/cplex_interface.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cplex/nl_cplex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cplex/grad_cplex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cplex/cplex_params.cpp
    )

    add_library(solver_cplex STATIC ${CPLEX_SRCS})

    target_include_directories(solver_cplex
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/cplex
            ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils
            ${CPLEX_CPLEX_INCLUDE_DIR}
            ${CPLEX_CONCERT_INCLUDE_DIR}
    )

    target_link_libraries(solver_cplex
        PUBLIC
            solver_common
            ${ILOCPLEX_LIB}
            ${CPLEX_LIB}
            ${CONCERT_LIB}
            Threads::Threads
            ${_DL_LIB}
    )

    if(TARGET MPI::MPI_CXX)
        target_link_libraries(solver_cplex PUBLIC MPI::MPI_CXX)
    endif()

    target_compile_definitions(solver_cplex PUBLIC USE_CPLEX)
    if(NODEBUGPRINT)
        target_compile_definitions(solver_cplex PUBLIC NODEBUGPRINT)
    endif()
endif()

# ============================================================
# XPRESS variant: solver_xpress
# ============================================================
if(USE_XPRESS)
    if(NOT XPRESS_HOME)
        message(FATAL_ERROR "USE_XPRESS=ON but XPRESS_HOME not set. "
                "Point it at your Xpress installation root (or set XPRESSDIR env).")
    endif()

    # Find Xpress library
    set(_XPRS_LIB_DIRS
        "${XPRESS_HOME}/lib"
        "${XPRESS_HOME}/lib64"
    )

    find_library(XPRS_LIB NAMES xprs HINTS ${_XPRS_LIB_DIRS} REQUIRED)

    message(STATUS "Xpress lib   : ${XPRS_LIB}")
    message(STATUS "Xpress inc   : ${XPRESS_HOME}/include")

    # XPRESS sources
    set(XPRESS_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/xpress/xpress_interface.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/xpress/nl_xpress.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/xpress/grad_xpress.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/xpress/xpress_params.cpp
    )

    add_library(solver_xpress STATIC ${XPRESS_SRCS})

    target_include_directories(solver_xpress
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/xpress
            ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils
            ${XPRESS_HOME}/include
    )

    target_link_libraries(solver_xpress
        PUBLIC
            solver_common
            ${XPRS_LIB}
    )

    if(TARGET MPI::MPI_CXX)
        target_link_libraries(solver_xpress PUBLIC MPI::MPI_CXX)
    endif()

    target_compile_definitions(solver_xpress PUBLIC USE_XPRESS)
    if(NODEBUGPRINT)
        target_compile_definitions(solver_xpress PUBLIC NODEBUGPRINT)
    endif()
endif()

# ============================================================
# HiGHS variant: solver_highs
# ============================================================
if(USE_HIGHS)
    if(NOT HIGHS_HOME)
        message(FATAL_ERROR "USE_HIGHS=ON but HIGHS_HOME not set. "
                "Point it at the prefix holding include/highs and lib/cmake/highs.")
    endif()

    find_package(highs CONFIG REQUIRED
        PATHS "${HIGHS_HOME}" NO_DEFAULT_PATH)
    message(STATUS "HiGHS        : ${highs_DIR} (version ${highs_VERSION})")

    set(HIGHS_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/highs/highs_interface.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/highs/nl_highs.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/highs/grad_highs.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/highs/highs_params.cpp
    )

    add_library(solver_highs STATIC ${HIGHS_SRCS})

    target_include_directories(solver_highs
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/highs
            ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils
    )

    target_link_libraries(solver_highs
        PUBLIC
            solver_common
            highs::highs
    )

    if(TARGET MPI::MPI_CXX)
        target_link_libraries(solver_highs PUBLIC MPI::MPI_CXX)
    endif()

    target_compile_definitions(solver_highs PUBLIC USE_HIGHS)
    if(NODEBUGPRINT)
        target_compile_definitions(solver_highs PUBLIC NODEBUGPRINT)
    endif()
endif()

# ============================================================
# Sanity check
# ============================================================
if(NOT USE_GUROBI AND NOT USE_CPLEX AND NOT USE_XPRESS AND NOT USE_HIGHS)
    message(STATUS "solver_interfaces: no solver backend is ON; only solver_common will be built.")
endif()
