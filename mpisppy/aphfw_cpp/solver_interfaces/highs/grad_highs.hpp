///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#ifndef GRAD_HIGHS_HPP
#define GRAD_HIGHS_HPP

/*
The so-called MILP solve for FW (based on the gradient, hence "grad").
The model's objective is linear (HiGHSInterface refuses a quadratic one), so
the gradient objective is linear too and the solve is a MIP.
*/

#include "highs_interface.hpp"

class HiGHSGrad : public HiGHSInterface {
public:
  HiGHSGrad(const std::string& scenarioName, const std::string& logFileName);
  virtual ~HiGHSGrad() = default;
  // can by moved, but not copied
  HiGHSGrad(const HiGHSGrad&) = delete;
  HiGHSGrad& operator=(const HiGHSGrad&) = delete;
  HiGHSGrad(HiGHSGrad&&) = default;
  HiGHSGrad& operator=(HiGHSGrad&&) = default;

  const std::vector<double>& getGradientAtReference() const {
    return gradientAtReference;
  }
  void updateReferenceVector(const std::vector<double>& referenceX); // does a lot
  bool addObjectiveCutoff(double cutoffValue) override;
  void removeObjectiveCutoff() override;

private:
  std::vector<double> gradientAtReference;
  HighsInt cutoffRow_ = -1;   // -1: no cutoff row
};
#endif  // GRAD_HIGHS_HPP
