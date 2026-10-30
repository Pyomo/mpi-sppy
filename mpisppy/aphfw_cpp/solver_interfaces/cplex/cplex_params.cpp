///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "cplex_params.hpp"

CPLEXParams::CPLEXParams(const std::string &jsonFileName)
  : SolverParams(jsonFileName)
{
  extrasWarn(canHave, "CPLEXParams constructor will not process extras: ");
}

void CPLEXParams::pushSolverParams(SolverInterface& scenario) {
  // Dynamic cast to a CPLEXInterface. If scenario is not CPLEX, skip
  CPLEXInterface* s = dynamic_cast<CPLEXInterface*>(&scenario);
  if (!s) {
    // Not a CPLEX solver instance — nothing to do.
    return;
  }

  if (auto v = getMipGap()) {
    const double sv = *v;
    // Use the NumParam overload
    s->setCplexParam(IloCplex::Param::MIP::Tolerances::MIPGap, sv);
  }
  if (auto v = getTimeLimit()) {
    const double sv = *v;
    // TimeLimit may be named TiLim in some versions; this compiles for common Concert versions.
    s->setCplexParam(IloCplex::Param::TimeLimit, sv);
  }
  if (auto v = getThreads()) {
    const int sv = *v;
    s->setCplexParam(IloCplex::Param::Threads, sv);
  }
}
