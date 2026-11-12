///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "gurobi_params.hpp"

GurobiParams::GurobiParams(const std::string jsonFileName):
  SolverParams(jsonFileName)
{
  extrasWarn(canHave, "GurobiParams constructor will not process extras: ");
}

void GurobiParams::pushSolverParams(SolverInterface& scenario) {
  GurobiInterface* s = dynamic_cast<GurobiInterface*>(&scenario);
  // tbd: templatize
  if (auto v = getMipGap()) {
    const double sv = v.value();
    s->setGurobiParam(GRB_DoubleParam_MIPGap, sv);
  }
  if (auto v = getTimeLimit()) {
    const double sv = v.value();
    s->setGurobiParam(GRB_DoubleParam_TimeLimit, sv);
  }
  if (auto v = getThreads()) {
    const int sv = v.value();
    s->setGurobiParam(GRB_IntParam_Threads, sv);
  }
}
