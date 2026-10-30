///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
// src/FWAPH/appRun.cpp
#include "appRun.hpp"

#include "FWAPH.hpp"
#include "ProblemWithSolvers.hpp"
#include "groupController.hpp"
#include "scenario.hpp"
#include "utils/timer.hpp"
#include "config/config.hpp"
#include "CLI11.hpp"

int runFWAPHWithArgs(int argc, char** argv, MPI_Comm comm)
{
  int rank;  // to pass to the timer
  MPI_Comm_rank(comm, &rank);

  GlobalTimer gt(rank); 
  install_global_timer(&gt);
  global_toc("Start");

  FWAPH controller;
  controller.setupComm(comm);

  CLI::App app{"FW APH"};
  Config config(app);
  FWAPH::registerOptions(config);
  groupController<Scenario>::registerOptions(config);  // static member of templated class

  CLI11_PARSE(app, argc, argv);

  if (controller.getRank() == 0) {
    std::cout << std::endl;
    config.pprint(std::cout);
    std::cout << std::endl;
  }

  controller.setup(config);

  global_toc("Start run");
  controller.run();         // in groupController

  global_toc("Run complete");

  controller.printResults();
  controller.printRunStatistics();

  controller.saveWTrace();  // do nothing if WTrace not selected
    
  global_toc("Finish.");

  // return 0 for success if we got this far
  return 0;
}
