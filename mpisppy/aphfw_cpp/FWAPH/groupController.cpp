///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
// originally contributed by Jonathan Eckstein

// CPP file for generic multithreading + overlaid asynchronous MPI
// communication work dispatcher.  The only code here is the little widget to
// invoke performTask with an exposed class pointer.


#include "groupController.hpp"


// Helper function to call subdomain::performTask with an exposed class pointer, so
// that it can be invoked without implicit class context from std::async 

void runTask(subdomain* ptr)
{
        ptr->performTask();
}
