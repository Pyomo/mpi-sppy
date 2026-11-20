///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "nl_highs.hpp"
#include <limits>

HiGHSNL::HiGHSNL(const std::string& scenarioName,
                 const std::string& logFileName):
  HiGHSInterface(scenarioName, logFileName)
{
  // Note that the indexes in the nonant maps still refer to x values because g is appended
  xSize = numVars;   // net of g vars that will be added later
  const HighsInt numRow = highs_->getNumRow();
  if (numRow > 0) {
    highsCheck(highs_->deleteRows(0, numRow - 1), "deleteRows");
  }
  if (xSize > 0 && !highs_->getLp().integrality_.empty()) {
    std::vector<HighsVarType> continuous(xSize, HighsVarType::kContinuous);
    highsCheck(highs_->changeColsIntegrality(0, xSize - 1, continuous.data()),
               "changeColsIntegrality");
  }
}

void HiGHSNL::addANewr(const std::vector<double>& r) {
  // add r to R and update the NL model (i.e. append things)
  if (static_cast<int>(r.size()) != xSize) {
    throw std::runtime_error("addANewr: r.size()=" + std::to_string(r.size()) +
                             " != xSize=" + std::to_string(xSize));
  }
  appendR(r);

  const HighsInt gCol = highs_->getNumCol();
  if (numRows_ == 0) {
    // first call: the g column, then the rows that use it
    highsCheck(highs_->addCol(0.0, 0.0, 1.0, 0, nullptr, nullptr), "addCol (first g)");
    fullObjectiveRebuild();  // the Hessian has to cover the new column

    const HighsInt nrows = 1 + xSize;
    std::vector<double> rhs(nrows, 0.0);
    rhs[0] = 1.0;  // sum-to-one
    std::vector<HighsInt> starts;
    std::vector<HighsInt> index;
    std::vector<double> value;
    starts.reserve(nrows);
    index.reserve(1 + 2 * xSize);
    value.reserve(1 + 2 * xSize);
    starts.push_back(0);   // row 0: g
    index.push_back(gCol);
    value.push_back(1.0);
    for (int i = 0; i < xSize; ++i) {  // rows 1..xSize: x_i - r_i g
      starts.push_back(static_cast<HighsInt>(index.size()));
      index.push_back(i);
      value.push_back(1.0);
      index.push_back(gCol);
      value.push_back(-r[i]);
    }
    highsCheck(highs_->addRows(nrows, rhs.data(), rhs.data(),
                               static_cast<HighsInt>(index.size()),
                               starts.data(), index.data(), value.data()),
               "addRows (first)");
    numRows_ = nrows;
  } else {
    // later calls: a g column with its coefficients in the existing rows
    std::vector<HighsInt> index(1 + xSize);
    std::vector<double> value(1 + xSize);
    index[0] = 0;
    value[0] = 1.0;
    for (int i = 0; i < xSize; ++i) {
      index[1 + i] = 1 + i;
      value[1 + i] = -r[i];
    }
    highsCheck(highs_->addCol(0.0, 0.0, 1.0, 1 + xSize, index.data(), value.data()),
               "addCol (g)");
    fullObjectiveRebuild();
  }
}
