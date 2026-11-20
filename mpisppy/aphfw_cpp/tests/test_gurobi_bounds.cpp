///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "solver_interfaces/gurobi/gurobi_interface.hpp"

namespace {

// Exposes whether Gurobi still holds the last solution: any model change
// discards it, and with it the MIP start the LMO solves rely on.
class GurobiBoundsProbe : public GurobiInterface {
public:
  using GurobiInterface::GurobiInterface;
  bool hasSolution() {
    try {
      vars[0].get(GRB_DoubleAttr_X);
      return true;
    } catch (const GRBException&) {
      return false;
    }
  }
};

}  // namespace

// The mpi-sppy hub pushes -inf/+inf for an unbounded nonant every cycle.
// Pushing the same infinite bounds again must not count as a change.
TEST(GurobiBounds, UnchangedInfiniteBoundsKeepSolution) {
  GurobiBoundsProbe gi("smallFWAPH/mps_dir/scen0", "");
  const int n = gi.getTotalNonantCount();
  ASSERT_GT(n, 0);
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> lb(n, -inf), ub(n, inf);

  gi.setNonantBounds(lb, ub);  // a real change from the file's bounds
  gi.solve();
  ASSERT_TRUE(gi.hasSolution());

  gi.setNonantBounds(lb, ub);  // the same bounds again
  EXPECT_TRUE(gi.hasSolution());
}
