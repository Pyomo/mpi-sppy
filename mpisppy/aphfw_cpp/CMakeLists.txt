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

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AphfwSettings.cmake)

add_subdirectory(utils)
add_subdirectory(solver_interfaces)
add_subdirectory(FWAPH)
add_subdirectory(python)
