///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "nl_xpress.hpp"
#include <numeric>

XpressNL::XpressNL(const std::string& scenarioName,
                   const std::string& logFileName)
  : XpressInterface(scenarioName, logFileName)
{
  xSize = numVars;  // save original var count before adding g vars
  dropAllConstraints();
  relaxIntegrality();
  isMIP_ = false;  // NL is always continuous
}

void XpressNL::dropAllConstraints() {
  int nrows = 0;
  xprsCheck(XPRSgetintattrib(prob_, XPRS_ROWS, &nrows),
            "XPRSgetintattrib(ROWS)");

  if (nrows > 0) {
    std::vector<int> rowind(nrows);
    std::iota(rowind.begin(), rowind.end(), 0);
    xprsCheck(XPRSdelrows(prob_, nrows, rowind.data()),
              "XPRSdelrows");
  }
  numRows_ = 0;
}

void XpressNL::relaxIntegrality() {
  // Change all variable types to continuous
  std::vector<int> colind(xSize);
  std::iota(colind.begin(), colind.end(), 0);
  std::vector<char> coltype(xSize, 'C');
  xprsCheck(XPRSchgcoltype(prob_, xSize, colind.data(), coltype.data()),
            "XPRSchgcoltype");
}

void XpressNL::addANewr(const std::vector<double>& r) {
  if (static_cast<int>(r.size()) != xSize) {
    throw std::runtime_error("addANewr: r.size()=" + std::to_string(r.size()) +
                             " != xSize=" + std::to_string(xSize));
  }

  appendR(r);

  // Get the current total number of columns (x vars + existing g vars)
  int totalCols = 0;
  xprsCheck(XPRSgetintattrib(prob_, XPRS_COLS, &totalCols),
            "XPRSgetintattrib(COLS) in addANewr");

  if (numRows_ == 0) {
    // ---- First call: no rows exist yet ----
    // Add g column with no row coefficients (rows don't exist yet)
    double gobj = 0.0;
    double glb = 0.0, gub = 1.0;
    int cstart[] = {0};
    xprsCheck(XPRSaddcols(prob_, 1, 0, &gobj, cstart, nullptr, nullptr, &glb, &gub),
              "XPRSaddcols (first g)");

    int gCol = totalCols;  // the new g column index

    // Rebuild objective (accounts for x variables only; g has cost 0)
    fullObjectiveRebuild();

    // Now add rows: row 0 = sum-to-one, rows 1..xSize = convex combination
    int nrows = 1 + xSize;
    int totalNZ = 1 + 2 * xSize;  // row 0: 1 nz, each combo row: 2 nz

    std::vector<char> rowtype(nrows, 'E');   // all equality
    std::vector<double> rhs(nrows, 0.0);
    rhs[0] = 1.0;  // sum-to-one

    std::vector<int> starts;
    std::vector<int> colind;
    std::vector<double> rowcoef;
    starts.reserve(nrows);
    colind.reserve(totalNZ);
    rowcoef.reserve(totalNZ);

    // Row 0: g_col with coefficient 1
    starts.push_back(0);
    colind.push_back(gCol);
    rowcoef.push_back(1.0);

    // Rows 1..xSize: x_i with coef 1, g_col with coef -r[i]
    for (int i = 0; i < xSize; ++i) {
      starts.push_back(static_cast<int>(colind.size()));
      colind.push_back(i);     // x_i
      rowcoef.push_back(1.0);
      colind.push_back(gCol);  // g
      rowcoef.push_back(-r[i]);
    }

    xprsCheck(XPRSaddrows(prob_, nrows, totalNZ, rowtype.data(), rhs.data(),
                           nullptr, starts.data(), colind.data(), rowcoef.data()),
              "XPRSaddrows (first)");

    numRows_ = nrows;

  } else {
    // ---- Subsequent calls: rows exist ----
    // Add g column WITH row coefficients for existing rows
    int ncoefs = 1 + xSize;
    double gobj = 0.0;
    double glb = 0.0, gub = 1.0;

    std::vector<int> rowind(ncoefs);
    std::vector<double> rowcoefvec(ncoefs);

    rowind[0] = 0;           // sum-to-one row
    rowcoefvec[0] = 1.0;

    for (int i = 0; i < xSize; ++i) {
      rowind[1 + i] = 1 + i;       // combination row
      rowcoefvec[1 + i] = -r[i];
    }

    int cstart[] = {0};
    xprsCheck(XPRSaddcols(prob_, 1, ncoefs, &gobj, cstart,
                           rowind.data(), rowcoefvec.data(), &glb, &gub),
              "XPRSaddcols (subsequent g)");

    // Rebuild objective
    fullObjectiveRebuild();
  }
}

std::vector<double> XpressNL::copyEntireSolution() const {
  // Return only the original x variables (not g variables)
  // Fetch fresh solution from solver
  int ncols = 0;
  XPRSgetintattrib(prob_, XPRS_COLS, &ncols);
  std::vector<double> fullSol(ncols);
  XPRSgetlpsol(prob_, fullSol.data(), nullptr, nullptr, nullptr);

  std::vector<double> returnVector(numVars);  // numVars == xSize for NL after parent ctor
  for (int i = 0; i < xSize; ++i) {
    returnVector[i] = fullSol[i];
  }
  // Also update the parent's solution_ so extractNonantSolution works
  solution_.resize(ncols);
  std::copy(fullSol.begin(), fullSol.end(), solution_.begin());
  return returnVector;
}
