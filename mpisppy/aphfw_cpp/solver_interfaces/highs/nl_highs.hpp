///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#ifndef NL_HIGHS_HPP
#define NL_HIGHS_HPP
/*
The so-called NL solve for FW; see nl_gurobi.hpp for the formulation.  The x
vars keep their bounds, lose their integrality and every constraint, and each
r adjoined to R brings a weight g in [0,1] with
  sum_{r in R} g_r = 1   (row 0)
  x_i - sum_{r in R} r_i g_r = 0   (row 1 + i)
so the solve is a continuous convex QP, which HiGHS can do.
 */
// Note: Upper case R is a vector of vectors and r is a member (a vector)

#include "highs_interface.hpp"

class HiGHSNL : public HiGHSInterface {
public:
  HiGHSNL(const std::string& scenarioName, const std::string& logFileName);
  ~HiGHSNL() override = default;
  // can by moved, but not copied
  HiGHSNL(const HiGHSNL&) = delete;
  HiGHSNL& operator=(const HiGHSNL&) = delete;
  HiGHSNL(HiGHSNL&&) = default;
  HiGHSNL& operator=(HiGHSNL&&) = default;

  void addANewr(const std::vector<double>& r);  // also update obj
  // copyEntireSolution needs no override: it returns the first numVars
  // columns, which are the x vars.

  std::vector<std::unique_ptr<const std::vector<double>>>& get_R() {return R;}

private:
  std::vector<std::unique_ptr<const std::vector<double>>> R;
  void appendR(const std::vector<double>& r) {
    R.push_back(std::make_unique<const std::vector<double>>(r));
  }
  int xSize = 0;    // without the g vars
  int numRows_ = 0;
};
#endif  // NL_HIGHS_HPP
