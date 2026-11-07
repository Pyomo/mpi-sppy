///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "grad_cplex.hpp"

CPLEXGrad::CPLEXGrad(const std::string& scenarioName)
  : CPLEXGrad(scenarioName, std::string())   // delegate
{}


CPLEXGrad::CPLEXGrad(const std::string& scenarioName, const std::string& logFileName)
  : CPLEXInterface(scenarioName, logFileName)
{
  const IloInt n = vars.getSize();
  gradientAtReference.assign(static_cast<size_t>(n), 0.0);
  referenceX.assign(static_cast<size_t>(n), 0.0);
}


void CPLEXGrad::updateReferenceVector(const std::vector<double>& refX)
{
  // Seed the next MIP solve from the current incumbent (mirrors GurobiGrad)
  setStartAtCurrent();

  // store full-size reference vector (same ordering as vars / fullC)
  this->referenceX = refX;

  const size_t nFull = fullC.size();
  const IloInt nVars = vars.getSize();

  if (static_cast<size_t>(nVars) != nFull) {
    std::cerr << "  nVars="+std::to_string(nVars) + " nFull="+std::to_string(nFull) << std::endl;
    throw std::runtime_error("CPLEXGrad::updateReferenceVector: fullC / vars size mismatch");
  }
  if (this->referenceX.size() != nFull) {
    throw std::runtime_error("CPLEXGrad::updateReferenceVector: referenceX size mismatch");
  }
  if (!WPtr || !zPtr) {
    throw std::runtime_error("CPLEXGrad::updateReferenceVector: W or z not attached");
  }
  if (rho.size() != static_cast<size_t>(totalNonantCount)) {
    throw std::runtime_error("CPLEXGrad::updateReferenceVector: rho size unexpected");
  }

  IloExpr objExpr = objective.getExpr();

  for (size_t i = 0; i < nFull; ++i) {
    // base gradient = original objective coefficient
    double g = fullC[i];

    // map solver var index -> nonant index (size_t(-1) if not a nonant)
    const size_t nai = iToNai(static_cast<int>(i));
    if (nai != static_cast<size_t>(-1)) {
      g += (*WPtr)[nai]
           - rho[nai] * (*zPtr)[nai]
           + rho[nai] * this->referenceX[i];
    }

    gradientAtReference[i] = g;
    objExpr.setLinearCoef(vars[static_cast<IloInt>(i)], g);
  }

  objective.setExpr(objExpr);
  objExpr.end();
}

bool CPLEXGrad::addObjectiveCutoff(double cutoffValue) {
  if (hasCutoffConstr_) removeObjectiveCutoff();
  IloExpr lhs(env);
  const IloInt nVars = vars.getSize();
  for (IloInt i = 0; i < nVars; ++i) {
    if (gradientAtReference[static_cast<size_t>(i)] != 0.0) {
      lhs += gradientAtReference[static_cast<size_t>(i)] * vars[i];
    }
  }
  objCutoffConstr_ = IloRange(env, -IloInfinity, lhs, cutoffValue, "fw_obj_cutoff");
  model.add(objCutoffConstr_);
  lhs.end();
  hasCutoffConstr_ = true;
  return true;
}

void CPLEXGrad::removeObjectiveCutoff() {
  if (hasCutoffConstr_) {
    model.remove(objCutoffConstr_);
    objCutoffConstr_.end();
    hasCutoffConstr_ = false;
  }
}
