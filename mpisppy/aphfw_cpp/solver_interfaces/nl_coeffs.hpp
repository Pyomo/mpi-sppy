///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#ifndef NL_COEFFS_HPP
#define NL_COEFFS_HPP
/* Create, and perhaps update, the coefficients for NL
(If there is ever an NL class, this should probably migrate to it)
This is obsolete because we switched to the extra variables, simpler formulation
*/

#include <map>
#include "solver_interface.hpp"

class NLCoeffs {
public:
  NLCoeffs(SolverInterface& SI,
           std::vector<std::unique_ptr<const std::vector<double>>>& R);
  ~NLCoeffs() {}
  double getLinCoeff(int j) const {return linCoeffs.at(j);}
  double getQuadCoeff(int j) const {return quadCoeffs.at(j);}
  double getCrossCoeff(int j, int k) const {return crossCoeffs.at({j,k});}
  int getN() const { return linCoeffs.size(); }
private:
  std::map<int, double> linCoeffs;
  std::map<int, double> quadCoeffs;
  std::map<std::tuple<int,int>, double> crossCoeffs;
};


#endif
