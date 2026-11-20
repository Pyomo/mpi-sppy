///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#ifndef GRAD_GUROBI_HPP
#define GRAD_GUROBI_HPP

/*
The so-called MILP solve for FW (based on the gradient, hence "grad").
*/

#include "gurobi_interface.hpp"

class GurobiGrad : public GurobiInterface {
public:
  GurobiGrad(const std::string& scenarioName, const std::string& logFileName);
  virtual ~GurobiGrad() = default;
  // can by moved, but not copied
  GurobiGrad(const GurobiGrad&) = delete;
  GurobiGrad& operator=(const GurobiGrad&) = delete;
  GurobiGrad(GurobiGrad&&) = default;
  GurobiGrad& operator=(GurobiGrad&&) = default;
      
  const std::vector<double>& getGradientAtReference() const {
    return gradientAtReference;
  }
  void updateReferenceVector(const std::vector<double>& referenceX); // does a lot
  bool addObjectiveCutoff(double cutoffValue) override;
  void removeObjectiveCutoff() override;

private:
  const std::vector<double> referenceX;
  std::vector<double> gradientAtReference;
  // The quadratic part of the original objective, kept aside at construction.
  // The gradient objective replaces the linear part and keeps this one, so
  // each update rebuilds from here instead of adding to what it built last
  // time; see the note in updateReferenceVector.
  GRBQuadExpr baseQuadObj_;
  GRBConstr cutoffConstr_;
  GRBQConstr cutoffQConstr_;
  bool cutoffIsQuadratic_ = false;
  bool hasCutoffConstr_ = false;
};
#endif  // GRAD_GUROBI_HPP
