///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#ifndef GUROBI_INTERFACE_HPP
#define GUROBI_INTERFACE_HPP

/* Troubles with gurobi and cplex (notes from 1 Feb 2025):
  With cplex, I can't set quad coefficients directly and with
  gurobi I don't seem to be a able to set either directly.
  (and maybe for cplex, I could ask for the quad term coefficient,
  rather than track it)

  Later in Feb: I need to make model a smart pointer because it
  does not have default constructor and dev.start() needs to be
  called before it is constructed (so it can't be in the initialize list)

  - I think that to have log files in gurobi, some things will need to
  be different than for cplex because it is really a property of the env 
  and must be set before any models are created. We take the 
  logfile name  as a constructor arg and set it in the constructor.

 */

/* Overloaded words: "variable" and "index"

I use the "var" to refer to solver (e.g. gurobi) objects for model
variables.  I use the word "index" to refer to the position in the
list of all solver vars. This list is called "vars." However, when I
say "bound by index" I am mean it in the Fortran-66 sense, so I am
using the word index here to refer to the position in a vector or
list.

The vars for nonanticaptive variables are refered to as "nonants".
Maps that refer to nonant indexes, refer to the index in the vars
list.

Note that nontant var values are supplied to the caller in a dense
vector via a copy operation.
The caller is expected to not know,
or care, about where the nonant values are in the solver's version of the
model. However, if for debugging or something, the caller wants to
know the name of a nonant var, they need to use the position in the
nonant vector (call it pos). They can then use the function
nonantPositionToName(pos) to get the variable name used in the
original mps file.

The values for dense vectors W, rho, and z are supplied by the caller. A
reference to W and z vectors should be obtained from the caller and then
the caller can change the values. To send the new values to the solver,
the caller should call something to update the objective. The values for rho need
to be swapped out using a setter call.
 */

#include <gurobi_c++.h>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <optional>
#include <unordered_set>
#include "../solver_interface.hpp"
#include "../json.hpp"
#include "../../utils/utils.hpp"

using NonantTreeNameMap = std::map<std::string, std::vector<std::string>>;
using NonantTreeIndexMap = std::map<std::string, std::vector<int>>;

//===============================================

class GurobiInterface : public SolverInterface {
public:
  GurobiInterface(const std::string& scenarioName, const std::string& logFileName);
  virtual ~GurobiInterface() override;
  // can be moved, but not copied
  GurobiInterface(const GurobiInterface&) = delete;
  GurobiInterface& operator=(const GurobiInterface&) = delete;
  GurobiInterface(GurobiInterface&&) = default;
  GurobiInterface& operator=(GurobiInterface&&) = default;
  
  objAndBnd solve() override;
  void interruptSolve() override;
  std::string getVarName(const size_t i) const override {
    return vars[i].get(GRB_StringAttr_VarName);
  }
  virtual std::vector<double> copyNonantSolution() const;
  virtual std::vector<double> copyEntireSolution() const;
  virtual void extractNonantSolution(std::vector<double>& nonantSolution) const override;
  virtual void extractNodeSolution(const std::string nodeName,
                           std::vector<double>& nodeSolution) const override;
  virtual void fullObjectiveRebuild() override;
  void applyBoundsToSolver() override;
  double getVarLB(int varIdx) const override;
  double getVarUB(int varIdx) const override;

  void setLogFile(const std::string fileName) override {
    std::cerr << "Warning: The log file cannot be set dynamically in Gurobi"
              << std::endl;
  }
  void logMsg(std::string msg) const override {
    std::cout << "Gubori log message to cout: " << msg << std::endl;
  }
  

  // we have to overload this, just like in cplex (could templatize)
  void setGurobiParam(const GRB_IntParam param, const std::any &value);
  void setGurobiParam(const GRB_DoubleParam param, const std::any &value);

  // Algorithmic data is all declared at the parent level
  
  // from here down are functions the caller should rarely, if ever, need.
  void printSolution() override;
  void nonantSolutionVectorToCsv(std::string fileName) const override;
  void exportModel(const std::string fname) override {  // have an mps file name suffix
    model->write(fname.c_str());
  }
  bool setStartAtCurrent() override; 
   
protected:
  // gurobi model objects
  GRBEnv env;
  std::unique_ptr<GRBModel> model;
  int numVars;  // we need this in Gurobi (should nl update it?)
  GRBVar* vars;   // c-style array
  GRBQuadExpr objExpr;  // might need to get it repeatedly?

  void _buildNonantsMaps();
  
  // let a child class (e.g. NL) start over
  void clearModel() override;

};
#endif // GUROBI_INTERFACE_HPP
