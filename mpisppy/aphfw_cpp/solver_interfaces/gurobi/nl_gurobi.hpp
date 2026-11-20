///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#ifndef NL_GUROBI_HPP
#define NL_GUROBI_HPP
// tbd: add a function to incrementally update the model for a new column
/*
Note: with considerable effort, one could factor the gurobi
and cplex version; there is effort to the inhertance from
solver_interface (you might need to have-a solver)

The so-called NL solve for FW.
We get a set of points, R. Starting with one point and growing one at a time.
So we will take R as a vector of vectors.

The optimization vars as x and g (for weight in German) with dimension |R| and
the only constraints are that across its dimensions it sums to one:
sum_{r \in R} g_{r} = 1
and that the x values are a convex combination:
x_i - \sum_{r \in R} rg_{r} = 0
(The r's are unique, so I can use them as indexes here... but not in the code.)

The constructor will be NOT be given a set R, or even one r because
the caller needs to assign W, rho, and etc. first. The the caller
can call addANewr.

 */
// Note: Upper case R is a vector of vectors and r is a member (a vector)

#include "gurobi_interface.hpp"
#include <numeric>
#include <algorithm>

class GurobiNL : public GurobiInterface {
public:
  GurobiNL(const std::string& scenarioName, const std::string& logFileName);
  ~GurobiNL() override;
  // can by moved, but not copied
  GurobiNL(const GurobiNL&) = delete;
  GurobiNL& operator=(const GurobiNL&) = delete;
  GurobiNL(GurobiNL&&) = default;
  GurobiNL& operator=(GurobiNL&&) = default;
  
  void addANewr(const std::vector<double>& r);  // also update obj
  // Note: we don't need to override most of the functions that get solutions because the x values are all there
  std::vector<double> copyEntireSolution() const override;
  
  std::vector<std::unique_ptr<const std::vector<double>>>& get_R() {return R;}  

private:
  std::vector<std::unique_ptr<const std::vector<double>>> R;
  void appendR(const std::vector<double>& r) {
    R.push_back(std::make_unique<const std::vector<double>>(r));
  }
  int xSize=0;  // without the g vars
};
#endif  // NL_GUROBI_HPP
