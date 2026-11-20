///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "highs_params.hpp"

HiGHSParams::HiGHSParams(const std::string jsonFileName):
  SolverParams(jsonFileName)
{
  extrasWarn(canHave, "HiGHSParams constructor will not process extras: ");
}

void HiGHSParams::pushSolverParams(SolverInterface& scenario) {
  HiGHSInterface* s = dynamic_cast<HiGHSInterface*>(&scenario);
  if (!s) {
    throw std::runtime_error("HiGHSParams::pushSolverParams: not a HiGHS interface");
  }
  if (auto v = getMipGap()) {
    s->setHiGHSOption("mip_rel_gap", v.value());
  }
  if (auto v = getTimeLimit()) {
    s->setHiGHSOption("time_limit", v.value());
  }
  if (auto v = getThreads()) {
    HiGHSInterface::requestThreads(v.value());
  }
}
