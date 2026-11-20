///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
// src/FWAPH/FWAPH_main.cpp
#include <mpi.h>
#include "appRun.hpp"
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rc = 0;
    try {
        // All the real work happens here
        rc = runFWAPHWithArgs(argc, argv, MPI_COMM_WORLD);
    }
    catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ERROR] Exception caught in main:\n" 
                  << "    " << e.what() << "\n" << std::endl;
        rc = 1;
    }
    catch (...) {
        std::cerr << "\n[CRITICAL ERROR] Unknown exception caught in main.\n" << std::endl;
        rc = 1;
    }

    MPI_Finalize();
    return rc;
}
