###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# tests/CMakeLists.txt
# Do NOT call cmake_minimum_required() or project() here.
# Those belong in the top-level CMakeLists.txt.

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# APHFW_CORE_TARGET and APHFW_MAIN_TARGET come from ../CMakeLists.txt.

add_executable(aphfw_main_test
    test_main_aph.cpp
    test_fw_export.cpp
)
if (USE_GUROBI)
    target_sources(aphfw_main_test PRIVATE test_numeric_retry.cpp test_gurobi_bounds.cpp)
endif()

target_include_directories(aphfw_main_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/FWAPH
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(aphfw_main_test
    PRIVATE
        ${APHFW_CORE_TARGET}
        aph_utils
        solver_common
        GTest::gtest
)

if (TARGET MPI::MPI_CXX)
    target_link_libraries(aphfw_main_test PRIVATE MPI::MPI_CXX)
endif()

include(GoogleTest)
# -------------------------------
# Register tests with CTest
# -------------------------------
include(GoogleTest)
# Discovery runs the test binary at build time, and its main() starts MPI,
# which on a cold machine (a fresh CI runner) took 5.1 s against CMake's
# default limit of 5.
gtest_discover_tests(
    aphfw_main_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DISCOVERY_TIMEOUT 60
)

# The startup W orthogonality check aborts via MPI_Abort rather than throwing
# (a node communicator need not span all ranks), so it cannot be a gtest case.
# Run the real executable on a problem whose two scenarios carry different rho
# and match the diagnostic. PASS_REGULAR_EXPRESSION decides the verdict, which
# is what we want given the process dies by abort.
add_test(
  NAME iter0_w_orthogonality_violation
  COMMAND ${APHFW_MAIN_TARGET}
          --problem-name smallRhoMismatchFWAPH/mps_dir
          --nl-json smallRhoMismatchFWAPH/nl.json
          --grad-json smallRhoMismatchFWAPH/grad.json
          --iter0-json smallRhoMismatchFWAPH/iter0.json
          --aph-max-cycles 2 --FW-tmax 4 --FW-tau 0.01
          --solver-log-file /dev/null
          --iter0-w-multiplier 1.0
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)
set_tests_properties(iter0_w_orthogonality_violation PROPERTIES
  TIMEOUT 120
  PASS_REGULAR_EXPRESSION "W ORTHOGONALITY VIOLATION at startup")

# Same data with the default multiplier: the guess is not used, so the check is
# skipped and the run must succeed. Guards against the check firing spuriously.
add_test(
  NAME iter0_w_orthogonality_skipped_when_off
  COMMAND ${APHFW_MAIN_TARGET}
          --problem-name smallRhoMismatchFWAPH/mps_dir
          --nl-json smallRhoMismatchFWAPH/nl.json
          --grad-json smallRhoMismatchFWAPH/grad.json
          --iter0-json smallRhoMismatchFWAPH/iter0.json
          --aph-max-cycles 2 --FW-tmax 4 --FW-tau 0.01
          --solver-log-file /dev/null
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)
set_tests_properties(iter0_w_orthogonality_skipped_when_off PROPERTIES
  TIMEOUT 120
  FAIL_REGULAR_EXPRESSION "W ORTHOGONALITY VIOLATION")

# diagCheckWInNPerp reduces over the node communicators, so every rank has to
# reach it. Guarding its calls with a rank-0 condition (as they once were) makes
# rank 0 wait in MPI_Allreduce while the others walk on, which no single-rank
# test can see. This runs two ranks at --verbosity 4, where the diagnostic is
# active, and is decided three ways:
#
#   TIMEOUT                 -- a deadlock never finishes, so it fails here
#   FAIL_REGULAR_EXPRESSION -- "^^^" marks a reported violation. W is genuinely
#                              in the complement for this data, so any such line
#                              is a false alarm (the rank-local version printed
#                              worstAbsMeanW=0.5 here).
#   PASS_REGULAR_EXPRESSION -- requires an OK line, so the test cannot pass by
#                              the diagnostic silently not running at all.
#
# The environment covers two things CI needs and a developer box does not.
# OMPI_ALLOW_RUN_AS_ROOT matches what the smoke scripts set, for containers
# running as root. The oversubscribe settings matter because the CI container
# advertises fewer slots than the two ranks requested, which Open MPI refuses by
# default ("There are not enough slots available in the system"); two tiny
# processes sharing a core is harmless here. Both spellings are given because
# the variable was renamed in Open MPI 5, and each version ignores the other's
# without complaint. They are environment variables rather than launcher flags
# so that a non-Open-MPI mpiexec ignores them instead of failing on an unknown
# option.
if (MPIEXEC_EXECUTABLE)
  add_test(
    NAME diag_nperp_multirank_no_deadlock
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:${APHFW_MAIN_TARGET}> ${MPIEXEC_POSTFLAGS}
            --problem-name smallFWAPH/mps_dir
            --nl-json smallFWAPH/nl.json
            --grad-json smallFWAPH/grad.json
            --iter0-json smallFWAPH/iter0.json
            --aph-max-cycles 3 --FW-tmax 4 --FW-tau 0.01
            --solver-log-file /dev/null
            --verbosity 4
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  )
  # The run takes about 2 seconds, so 60 is ample headroom on a slow runner
  # while keeping the cost of an actual deadlock down to a minute.
  set_tests_properties(diag_nperp_multirank_no_deadlock PROPERTIES
    TIMEOUT 60
    FAIL_REGULAR_EXPRESSION "DIAG-NPERP\\] \\^\\^\\^"
    PASS_REGULAR_EXPRESSION "DIAG-NPERP\\] Node=[^ ]+ OK"
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_rmaps_base_oversubscribe=1;PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe")
else()
  message(STATUS "MPIEXEC_EXECUTABLE not found; skipping diag_nperp_multirank_no_deadlock")
endif()

# The Python bindings, on the same small problem, against the module this build
# produced: PYTHONPATH points at its directory, so an installed aphfw_cpp cannot
# stand in for it. Each script exits non-zero on failure.
find_package(Python3 COMPONENTS Interpreter)
if (MPIEXEC_EXECUTABLE AND Python3_Interpreter_FOUND AND TARGET aphfw_cpp)
  foreach(_bindings rho_bridge omicron_bridge stepping)
    add_test(
      NAME bindings_${_bindings}
      COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:aphfw_cpp>
              ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              ${Python3_EXECUTABLE} -m mpi4py
              ${CMAKE_CURRENT_SOURCE_DIR}/bindings_${_bindings}.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    )
    set_tests_properties(bindings_${_bindings} PROPERTIES
      TIMEOUT 300
      ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_rmaps_base_oversubscribe=1;PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe")
  endforeach()
else()
  message(STATUS "mpiexec, Python or the aphfw_cpp target missing; skipping the bindings tests")
endif()
