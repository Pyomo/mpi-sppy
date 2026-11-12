///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#ifndef GUROBI_PARAMS_HPP
#define GUROBI_PARAMS_HPP

#include "../solver_params.hpp"
#include "gurobi_interface.hpp"

class GurobiParams : public SolverParams {
public:
  GurobiParams(const std::string jsonFileName);
  void pushSolverParams(SolverInterface& scenario) override;
};

#endif
