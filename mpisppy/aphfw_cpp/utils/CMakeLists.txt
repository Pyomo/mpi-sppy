###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# aphfw_cpp/utils/CMakeLists.txt

add_library(aph_utils
    timer.cpp
)

target_include_directories(aph_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(NODEBUGPRINT)
    target_compile_definitions(aph_utils PUBLIC NODEBUGPRINT)
endif()
