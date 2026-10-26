///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "xpress_params.hpp"

XpressParams::XpressParams(const std::string& jsonFileName)
  : SolverParams(jsonFileName)
{
  extrasWarn(canHave, "XpressParams constructor will not process extras: ");
}

void XpressParams::pushSolverParams(SolverInterface& scenario) {
  XpressInterface* s = dynamic_cast<XpressInterface*>(&scenario);
  if (!s) {
    return;  // Not an Xpress solver instance
  }

  if (auto v = getMipGap()) {
    s->setXpressDblControl(XPRS_MIPRELSTOP, *v);
  }
  if (auto v = getTimeLimit()) {
    s->setXpressDblControl(XPRS_TIMELIMIT, *v);
  }
  if (auto v = getThreads()) {
    s->setXpressIntControl(XPRS_THREADS, *v);
  }
}
