// appRun.hpp
#pragma once

#include <mpi.h>

// Runs the FWAPH application with given argc/argv.
// The MPI_Comm argument is currently ignored (we use MPI_COMM_WORLD),
// but we keep it in the signature in case we want a different communicator later.
// Assume MPI has already been initialized by the caller.
int runFWAPHWithArgs(int argc, char** argv, MPI_Comm comm);
