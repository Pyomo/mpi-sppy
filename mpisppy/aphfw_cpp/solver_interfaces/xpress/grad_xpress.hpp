///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#ifndef GRAD_XPRESS_HPP
#define GRAD_XPRESS_HPP

/*
The so-called MILP solve for FW (based on the gradient, hence "grad").
Xpress version.
*/

#include "xpress_interface.hpp"

class XpressGrad : public XpressInterface {
public:
  XpressGrad(const std::string& scenarioName, const std::string& logFileName);
  virtual ~XpressGrad() = default;

  // move-only
  XpressGrad(const XpressGrad&) = delete;
  XpressGrad& operator=(const XpressGrad&) = delete;
  XpressGrad(XpressGrad&&) = default;
  XpressGrad& operator=(XpressGrad&&) = default;

  const std::vector<double>& getGradientAtReference() const {
    return gradientAtReference;
  }
  void updateReferenceVector(const std::vector<double>& referenceX);
  bool addObjectiveCutoff(double cutoffValue) override;
  void removeObjectiveCutoff() override;

private:
  std::vector<double> referenceX;
  std::vector<double> gradientAtReference;
  int cutoffRowIndex_ = -1;
};

#endif  // GRAD_XPRESS_HPP
