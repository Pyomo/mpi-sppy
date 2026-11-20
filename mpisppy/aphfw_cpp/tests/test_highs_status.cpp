///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
// What HiGHSInterface::solve() reports when a solve does not end at an optimum,
// and the objective cutoff of HiGHSGrad.
//
// Fixtures, in highsStatus/, with X1 and X2 as the nonants:
//   infeasible  X1 + X2 >= 3 with both at most 1
//   unbounded   min -X1 - X2 with X1 + X2 >= 0
//   lp          min -3 X1 - 2 X2 s.t. X1 + X2 <= 4, X1 + 3 X2 <= 6, optimum
//               -12 at (4, 0); the origin is feasible
//   knapsack    a strongly correlated binary knapsack, optimum -1022; HiGHS
//               takes a node to prove it, after finding worse incumbents.
// The cutoff tests use scenario 0 of smallFWAPH: min X1 + 4 X2 + 9 X3 with
// X1 + X2 + X3 >= 1, all binary, optimum 1.

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "solver_interfaces/highs/grad_highs.hpp"
#include "solver_interfaces/highs/highs_interface.hpp"

namespace {

const double kInf = std::numeric_limits<double>::infinity();

// Sets the options that make a solve stop where the test needs it to.
class HiGHSStatusProbe : public HiGHSInterface {
public:
  using HiGHSInterface::HiGHSInterface;
  void option(const std::string& name, const std::string& value) {
    ASSERT_EQ(highs_->setOptionValue(name, value), HighsStatus::kOk) << name;
  }
  void option(const std::string& name, double value) {
    ASSERT_EQ(highs_->setOptionValue(name, value), HighsStatus::kOk) << name;
  }
  void option(const std::string& name, int value) {
    ASSERT_EQ(highs_->setOptionValue(name, value), HighsStatus::kOk) << name;
  }
  HighsModelStatus status() const { return highs_->getModelStatus(); }
};

std::string solveError(HiGHSInterface& solver) {
  try {
    solver.solve();
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

}  // namespace

TEST(HiGHSStatus, InfeasibleThrows) {
  HiGHSStatusProbe solver("highsStatus/infeasible", "");
  const std::string error = solveError(solver);
  EXPECT_NE(error.find("highsStatus/infeasible"), std::string::npos) << error;
  EXPECT_NE(error.find("nfeasible"), std::string::npos) << error;
  EXPECT_FALSE(solver.isInfeasibleDueToCutoff());
}

// With presolve off HiGHS can tell unbounded from infeasible.
TEST(HiGHSStatus, UnboundedThrows) {
  HiGHSStatusProbe solver("highsStatus/unbounded", "");
  solver.option("presolve", std::string("off"));
  const std::string error = solveError(solver);
  EXPECT_EQ(solver.status(), HighsModelStatus::kUnbounded);
  EXPECT_NE(error.find("is unbounded"), std::string::npos) << error;
}

// A continuous solve stopped early has a feasible point and no bound.  Primal
// simplex starts from the feasible origin, and stays there with no iterations
// allowed.
TEST(HiGHSStatus, ContinuousStoppedEarlyHasNoBound) {
  HiGHSStatusProbe solver("highsStatus/lp", "");
  solver.option("presolve", std::string("off"));
  solver.option("solver", std::string("simplex"));
  solver.option("simplex_strategy", 4);
  solver.option("simplex_iteration_limit", 0);
  const objAndBnd result = solver.solve();
  EXPECT_EQ(solver.status(), HighsModelStatus::kIterationLimit);
  EXPECT_EQ(result.bnd, -kInf);
  EXPECT_GT(result.obj, -12.0);
  EXPECT_FALSE(solver.hasNumericalBailout());
}

// A MIP stopped by its time limit, holding the last optimum as its start: the
// incumbent is reported, with a bound that does not pass it.
TEST(HiGHSStatus, MIPStoppedEarlyKeepsItsIncumbent) {
  HiGHSStatusProbe solver("highsStatus/knapsack", "");
  const objAndBnd optimal = solver.solve();
  EXPECT_EQ(solver.status(), HighsModelStatus::kOptimal);

  solver.setStartAtCurrent();
  solver.option("time_limit", 1e-9);
  const objAndBnd result = solver.solve();
  EXPECT_EQ(solver.status(), HighsModelStatus::kTimeLimit);
  EXPECT_DOUBLE_EQ(result.obj, optimal.obj);
  EXPECT_LE(result.bnd, result.obj);
  EXPECT_FALSE(solver.hasNumericalBailout());
}

// Stopped before finding anything: an error, not a result.
TEST(HiGHSStatus, StoppedWithoutASolutionThrows) {
  HiGHSStatusProbe solver("highsStatus/knapsack", "");
  solver.option("presolve", std::string("off"));
  solver.option("time_limit", 1e-9);
  const std::string error = solveError(solver);
  EXPECT_NE(error.find("and no solution"), std::string::npos) << error;
}

// Any other stop with a solution: use it, and raise the bailout flag.  Every
// objective coefficient is negative, so any nonzero incumbent meets this target.
TEST(HiGHSStatus, OtherStopWithSolutionRaisesBailout) {
  HiGHSStatusProbe solver("highsStatus/knapsack", "");
  solver.option("presolve", std::string("off"));
  solver.option("objective_target", 0.0);
  const objAndBnd result = solver.solve();
  EXPECT_EQ(solver.status(), HighsModelStatus::kObjectiveTarget);
  EXPECT_TRUE(solver.hasNumericalBailout());
  EXPECT_GT(result.obj, -1022.0);  // stopped short of the optimum
  EXPECT_LT(result.bnd, -1022.0);
}

class HiGHSCutoff : public ::testing::Test {
protected:
  HiGHSCutoff() : grad("smallFWAPH/mps_dir/scen0", ""), W(3, 0.0), z(3, 0.0) {
    grad.attach_W(W);
    grad.attach_z(z);
    grad.set_rho(std::vector<double>(3, 1.0));
    // At the origin, with W and z zero, the gradient is the original cost.
    grad.updateReferenceVector(std::vector<double>(3, 0.0));
  }
  HiGHSGrad grad;
  std::vector<double> W, z;
};

// No point meets the cutoff, which is the "no improving direction" answer and
// not an error; without the row the model solves as before.
TEST_F(HiGHSCutoff, BelowTheOptimumIsInfeasibleNotAnError) {
  ASSERT_TRUE(grad.addObjectiveCutoff(0.5));
  EXPECT_NO_THROW(grad.solve());
  EXPECT_TRUE(grad.isInfeasibleDueToCutoff());

  grad.removeObjectiveCutoff();
  EXPECT_NEAR(grad.solve().obj, 1.0, 1e-9);
}

TEST_F(HiGHSCutoff, AboveTheOptimumChangesNothing) {
  ASSERT_TRUE(grad.addObjectiveCutoff(2.0));
  EXPECT_NEAR(grad.solve().obj, 1.0, 1e-9);
  EXPECT_FALSE(grad.isInfeasibleDueToCutoff());
}

// Once the row is gone an infeasible model is an error again.
TEST(HiGHSCutoffRemoved, InfeasibilityIsNoLongerTolerated) {
  HiGHSGrad grad("highsStatus/infeasible", "");
  grad.addObjectiveCutoff(100.0);
  EXPECT_NO_THROW(grad.solve());
  EXPECT_TRUE(grad.isInfeasibleDueToCutoff());
  grad.removeObjectiveCutoff();
  EXPECT_NE(solveError(grad), "");
}
