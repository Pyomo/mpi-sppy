///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "grad_highs.hpp"
#include <limits>

HiGHSGrad::HiGHSGrad(const std::string& scenarioName,
                     const std::string& logFileName):
  HiGHSInterface(scenarioName, logFileName)
{
  // gradient for the original problem
  gradientAtReference.assign(numVars, 0.0);
}

void HiGHSGrad::updateReferenceVector(const std::vector<double>& referenceX)
// grab the reference vector and update the objective function
// also update the gradient
// Another side-effect: the current solution is the start of the next solve
{
  if (static_cast<int>(referenceX.size()) != numVars) {
    throw std::runtime_error("HiGHSGrad::updateReferenceVector: referenceX size mismatch");
  }
  if (!WPtr || !zPtr) {
    throw std::runtime_error("HiGHSGrad::updateReferenceVector: W or z not attached");
  }
  for (size_t i=0; i < fullC.size(); ++i) {
    gradientAtReference[i] = fullC[i];
    size_t nai;  // nonant index
    if ((nai = iToNai(i)) != -1) {
      gradientAtReference[i] += (*WPtr)[nai] - rho[nai] * (*zPtr)[nai] + rho[nai] * referenceX[i];
    }
  }
  if (numVars > 0) {
    highsCheck(highs_->changeColsCost(0, numVars - 1, gradientAtReference.data()),
               "changeColsCost in updateReferenceVector");
  }
  setStartAtCurrent();
}

bool HiGHSGrad::addObjectiveCutoff(double cutoffValue) {
  // An explicit row, objective <= cutoffValue, as the Gurobi backend adds.
  if (cutoffRow_ >= 0) removeObjectiveCutoff();
  std::vector<HighsInt> index;
  std::vector<double> value;
  for (int i = 0; i < numVars; ++i) {
    if (gradientAtReference[i] != 0.0) {
      index.push_back(i);
      value.push_back(gradientAtReference[i]);
    }
  }
  highsCheck(highs_->addRow(-std::numeric_limits<double>::infinity(), cutoffValue,
                            static_cast<HighsInt>(index.size()),
                            index.data(), value.data()),
             "addRow (obj cutoff)");
  cutoffRow_ = highs_->getNumRow() - 1;
  tolerateInfeasible_ = true;
  infeasibleDueToCutoff_ = false;
  return true;
}

void HiGHSGrad::removeObjectiveCutoff() {
  if (cutoffRow_ >= 0) {
    highsCheck(highs_->deleteRows(cutoffRow_, cutoffRow_), "deleteRows (obj cutoff)");
    cutoffRow_ = -1;
  }
  tolerateInfeasible_ = false;
}
