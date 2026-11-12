///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
// Guard for the NUMERIC retry in GurobiInterface::solve().
//
// numericRetry/numeric_qp.mps is a Frank-Wolfe subproblem saved when an
// aircondSetup run at rho 0.003 died: Gurobi returned GRB_NUMERIC, solve()
// threw, and the MPI job aborted.  With NumericFocus=2 it solves, to
// 2997.4386617790656.  The failure depends on the thread count -- it
// reproduces at Gurobi's default (all cores) on a 16-thread machine, not at
// Threads=1 -- so on another machine this test may pass without exercising
// the retry.  It cannot fail spuriously: if the first solve succeeds, the
// objective is the same.

#include <gtest/gtest.h>

#include "solver_interfaces/gurobi/gurobi_interface.hpp"

TEST(GurobiNumericRetry, IllConditionedProxQPSolves) {
    GurobiInterface gi("numericRetry/numeric_qp", "");
    objAndBnd r{};
    ASSERT_NO_THROW(r = gi.solve());
    EXPECT_NEAR(r.obj, 2997.4386617790656, 1e-3);
}
