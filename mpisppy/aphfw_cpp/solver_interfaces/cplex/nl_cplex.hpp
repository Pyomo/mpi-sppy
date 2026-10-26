///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#ifndef NL_CPLEX_HPP
#define NL_CPLEX_HPP

/*
The so-called NL solve for FW.
We get a set of points, R. Starting with one point and growing one at a time.
So we will take R as a vector of vectors.

The optimization var is now g (for weight) with dimension |R| and
the only constraint is that across its dimensions it sums to one:
sum_{r \in R} g_{r} = 1
(The r's are unique, so I can use them as indexes here... but not in the code.)

We take the quadratic objective from PH/APH and replace every variable with
sum_{r \in R}r g_{r} 

The only bummer is expanding the terms that were originally x^2.  For a base
model that is linear, there will be one such term per nonant.

We will have small, simplicial constraint and a potentially very long objective.
Although while R is small, the objective after term collection will also be
pretty small and we are doing that collection in addANewr().

Getting to work with i as an index into the vars objects "list":
The linear objective coefficient for nonant vars is c[i]+w[pos]-z[pos]*rho[pos]
where pos is the position of var i in the dense list of nonants.
For other vars, it is just c[i]
The quadradtics are only for nonants with coefficient rho[i]/2

The constructor will be NOT be given a set R, or even one R because
the caller needs to assign W, rho, and etc. first. The the caller
can call addANewr.

 */
// Note: Upper case R is a vector of vectors and r is a member (a vector)

#include "cplex_interface.hpp"
#include "../nl_coeffs.hpp"
#include <numeric>
#include <algorithm>

class CPLEXNL : public CPLEXInterface {
public:
  CPLEXNL(const std::string& scenarioName);
  CPLEXNL(const std::string& scenarioName, const std::string& logFileName);
  virtual ~CPLEXNL() = default;
  // can by moved, but not copied
  CPLEXNL(const CPLEXNL&) = delete;
  CPLEXNL& operator=(const CPLEXNL&) = delete;
  CPLEXNL(CPLEXNL&&) = default;
  CPLEXNL& operator=(CPLEXNL&&) = default;
  
  void addANewr(const std::vector<double>& r);  // also update obj
  std::vector<double> copyEntireSolution() const override;
  std::vector<std::unique_ptr<const std::vector<double>>>& get_R() {return R;}

private:
  std::vector<std::unique_ptr<const std::vector<double>>> R;
  void appendR(const std::vector<double>& r) {
    R.push_back(std::make_unique<const std::vector<double>>(r));
  }
  // NL model handles 
  IloRange        sumToOne_;  // “sum g = 1” row (null until first build)
  IloRangeArray   combCons_;  // xSize rows: vars[i] - Σ r_j[i] g_j == 0
  IloNumVarArray  gvars_;     // the g-variables you append
  IloInt          xSize = 0;  // number of vars before g vars are added
  
};
#endif  // NL_CPLEX_HPP
