///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "gurobi_interface.hpp"
#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>
#include <exception>
#include <cmath>
#include <filesystem>

std::string gurobiExprToString(const GRBQuadExpr expr);  // forward

GurobiInterface::GurobiInterface(const std::string& scenarioName,
                                 const std::string& logFileName):
  SolverInterface(scenarioName),
  env(true)
{
  try {
    env.set(GRB_StringParam_LogFile, logFileName); // use "" for none
    if (logFileName != "") {
      env.set(GRB_IntParam_LogToConsole, 0);
    }
    
    env.start();
    model = std::make_unique<GRBModel>(env, scenarioName + ".mps");  
    numVars = model->get(GRB_IntAttr_NumVars);
    vars = model->getVars();  // pointer to a c array
    objExpr = model->getObjective();
  } catch (GRBException& e) {
    std::cerr << "Gurobi Error code = " << e.getErrorCode() << std::endl;
    std::cerr << "Gurobi Error message: " << e.getMessage() << std::endl;
    throw;
  } catch (...) {
    std::cerr << "An unexpected error occurred during model construction or optimization." << std::endl;
    throw;
  }
  // get the full, original cost vector using the original vars before we clear
  fullC.assign(numVars, 0.0);
  for (int i = 0; i < numVars; i++) {
    double coeff = vars[i].get(GRB_DoubleAttr_Obj);
    fullC[i] = coeff;
  }  
  // the parent class loads the nonants from json
  _buildNonantsMaps();

  // deal with quadratic terms in original model
  fullQDiag.assign(static_cast<size_t>(numVars), 0.0);

  unsigned int qsize = objExpr.size();
  for (unsigned int k = 0; k < qsize; ++k) {
    double coef = objExpr.getCoeff(k);
    GRBVar v1 = objExpr.getVar1(k);
    GRBVar v2 = objExpr.getVar2(k);

    // map vars to indices using name->index you already build in _buildNonantsMaps
    std::string n1 = v1.get(GRB_StringAttr_VarName);
    std::string n2 = v2.get(GRB_StringAttr_VarName);

    int idx1 = varNameToIndex.at(n1);
    int idx2 = varNameToIndex.at(n2);

    if (idx1 == idx2) {
      fullQDiag[static_cast<size_t>(idx1)] += coef;
      hasOrigQuad = true;
    } else {
      hasOrigQuad = true;
      hasOrigQuadCross = true;
    }
  }

  if (hasOrigQuadCross) {
    throw std::runtime_error("Original objective has cross quadratic terms x_i*x_j (i!=j), not supported yet.");
  }
  
}

GurobiInterface::~GurobiInterface() {
  delete[] vars;
};

void GurobiInterface::fullObjectiveRebuild() {
  try {
    // sanity
    if (numVars != static_cast<int>(fullC.size())) {
      std::cerr << "WARNING: Gurobi fullObjectiveRebuild: numVars=" << numVars
                << " != fullC.size()=" << fullC.size() << std::endl;
    }
    if (WOn && !WPtr) {
      throw std::runtime_error("Gurobi fullObjectiveRebuild: WOn=true but WPtr is null");
    }
    
    if (proxOn && !zPtr) {
      throw std::runtime_error("Gurobi fullObjectiveRebuild: proxOn=true but zPtr is null");
    }
    if (rho.size() != static_cast<size_t>(totalNonantCount)) {
      throw std::runtime_error("Gurobi fullObjectiveRebuild: rho size mismatch vs totalNonantCount");
    }
    if (WPtr && WPtr->size() != static_cast<size_t>(totalNonantCount)) {
      throw std::runtime_error("Gurobi fullObjectiveRebuild: W size mismatch vs totalNonantCount");
    }
    if (zPtr && zPtr->size() != static_cast<size_t>(totalNonantCount)) {
      throw std::runtime_error("Gurobi fullObjectiveRebuild: z size mismatch vs totalNonantCount");
    }

    // ensure the nonant indexes are valid
    for (int vidx : nonantDenseIndexes) {
      if (vidx < 0 || vidx >= numVars) {
        throw std::runtime_error("Gurobi fullObjectiveRebuild: nonant index out of range: " + std::to_string(vidx));
      }
    }

    // Build expression from scratch
    GRBQuadExpr expr;

    // baseline linear
    for (int j = 0; j < numVars; ++j) {
      double c = fullC[static_cast<size_t>(j)];
      if (c != 0.0) expr += c * vars[j];
    }

    // baseline diagonal quadratics (if any)
    if (!fullQDiag.empty()) {
      for (int j = 0; j < numVars; ++j) {
        double q = fullQDiag[static_cast<size_t>(j)];
        if (q != 0.0) expr += q * vars[j] * vars[j];
      }
    }

    // overlay nonant deltas and prox quadratics
    int pos = 0;
    for (int vidx : nonantDenseIndexes) {
      double delta = 0.0;
      if (WOn)   delta += (*WPtr)[pos];
      if (proxOn) delta -= rho[pos] * (*zPtr)[pos];

      if (delta != 0.0) expr += delta * vars[vidx];

      if (proxOn) expr += (rho[pos] / 2.0) * vars[vidx] * vars[vidx];
      ++pos;
    }

    model->setObjective(expr, GRB_MINIMIZE);
    model->update();

    rhoChanged = false;

  } catch (GRBException &e) {
    std::cerr << "Gurobi exception in fullObjectiveRebuild: " << e.getErrorCode()
              << ": " << e.getMessage() << std::endl;
    numericalBailout.store(true, std::memory_order_relaxed);
    return;
  } catch (const std::exception &e) {
    std::cerr << "std::exception in Gurobi fullObjectiveRebuild: " << e.what() << std::endl;
    numericalBailout.store(true, std::memory_order_relaxed);
    return;
  } catch (...) {
    std::cerr << "Unknown exception in Gurobi fullObjectiveRebuild()" << std::endl;
    numericalBailout.store(true, std::memory_order_relaxed);
    return;
  }
}


void GurobiInterface::applyBoundsToSolver() {
  // Touch only the bounds that change.  Any modification, even to the value a
  // bound already has, makes Gurobi discard the last solution, and that
  // solution is the MIP start GurobiGrad::updateReferenceVector sets.  The
  // mpi-sppy hub pushes every bound every cycle, usually unchanged, so writing
  // them all left no LMO solve a start.  Gurobi stores an infinite bound as
  // +-GRB_INFINITY (1e100), so the hub's +-inf is clamped to that before
  // comparing, or an unbounded nonant would count as changed every cycle.
  bool changed = false;
  for (int pos = 0; pos < totalNonantCount; ++pos) {
    int vidx = nonantDenseIndexes[pos];
    const double lb = std::max(nonantLowerBounds[pos], -GRB_INFINITY);
    const double ub = std::min(nonantUpperBounds[pos], GRB_INFINITY);
    if (vars[vidx].get(GRB_DoubleAttr_LB) != lb) {
      vars[vidx].set(GRB_DoubleAttr_LB, lb);
      changed = true;
    }
    if (vars[vidx].get(GRB_DoubleAttr_UB) != ub) {
      vars[vidx].set(GRB_DoubleAttr_UB, ub);
      changed = true;
    }
  }
  if (changed) model->update();
}

double GurobiInterface::getVarLB(int varIdx) const {
  return vars[varIdx].get(GRB_DoubleAttr_LB);
}

double GurobiInterface::getVarUB(int varIdx) const {
  return vars[varIdx].get(GRB_DoubleAttr_UB);
}


void GurobiInterface::_buildNonantsMaps() {
  // Called by the constructor only; maybe it should be a lambda...
  //   because it should NOT be called again.
  // It is a long, yet simple function.
  std::unordered_set<std::string> modelVarSet;   // model var names
  std::unordered_set<std::string> nonantSet;  // nonant var names
  
  // create various maps and a local set for error checking
  for (const auto& [ndn, nd] : treeNodeDataMap) {
    const auto& currentList = nd.nonAnts;
    nonantSet.insert(currentList.begin(), currentList.end());
    nodeNonantCounts[ndn] = currentList.size();
  }

  for (int i = 0; i < numVars; ++i) {
    auto vname = vars[i].get(GRB_StringAttr_VarName);
    varNameToIndex[vname] = i;
    if (nonantSet.count(vname)) {
      modelVarSet.insert(vname);  // local
      nonantNameToIndex[vname] = i;
      nonantIndexToName[i] = vname;
    }
  }

  for (const auto& [ndn, nd] : treeNodeDataMap) {
    const auto& currentList = nd.nonAnts;
    std::for_each(currentList.begin(), currentList.end(), [&](auto vname) {
      // nonantIndexes[ndn].push_back(nonantNameToIndex[vname]);
      auto it = nonantNameToIndex.find(vname);
      if (it == nonantNameToIndex.end()) {
        throw std::runtime_error("Nonant not in model: " + vname);
      }
      nonantIndexes[ndn].push_back(it->second);
      
    });
  }  
  // The thrown message sort of assumes construction in progress
  std::unordered_set<std::string> difference;
  for (auto &vname : nonantSet) {
    if (modelVarSet.find(vname) == modelVarSet.end()) { // If not in model
      difference.insert(vname);
    }
  }
  if (!difference.empty()) {
    std::cerr << "Your solver may have failed to instantiate!" << std::endl;
    std::cerr << "If you are sure it did instantiate, then the following message is relevant:" << std::endl;
    std::cerr << "The following variables were in the nonant json,"
              << " but not the model (assuming it instantiated)" << std::endl;
    for (auto &vname : difference) {
      std::cerr << vname << std::endl;
    }
    throw std::runtime_error("no solver model; or noant vs model var name mismatch for " \
                             + nonantFileName + " " + mpsFileName);
  } 
  // Build dense list of all nonant variable indexes
  nonantDenseIndexes.clear();
  for (const auto& [nd, indexList] : nonantIndexes) {
    nonantDenseIndexes.insert(nonantDenseIndexes.end(),
                              indexList.begin(), indexList.end());
  }

  // Map solver variable index -> position in dense nonant vectors (W, z, rho)
  varToNonant.clear();
  for (size_t nai = 0; nai < nonantDenseIndexes.size(); ++nai) {
    varToNonant[nonantDenseIndexes[nai]] = nai;
  }

  std::unordered_set<int> seen;
  for (int vidx : nonantDenseIndexes) {
    if (!seen.insert(vidx).second) {
      throw std::runtime_error("Duplicate variable index in nonantDenseIndexes: " + std::to_string(vidx));
    }
  }
  
}


objAndBnd GurobiInterface::solve() {
  double obj, bound;

  interrupted.store(false, std::memory_order_relaxed);
  if (abortRequested()) {
    interrupted.store(true, std::memory_order_relaxed);
    return objAndBnd{0.0, 0.0};
  }
  // Gurobi latches a terminate request, so interruptSolve() only fires while
  // this flag is up; the guard clears it on every exit path.
  struct SolveScope {
    std::atomic<bool>& flag;
    ~SolveScope() { flag.store(false, std::memory_order_relaxed); }
  } solveScope{solveInProgress};
  solveInProgress.store(true, std::memory_order_relaxed);

  try {
    model->optimize();

    // Check the optimization status.
    int status = model->get(GRB_IntAttr_Status);
    if (status == GRB_NUMERIC) {
      // Retry with more careful numerics before giving up.  A small rho makes
      // the proximal QPs ill-conditioned (prox coefficient 1e-3 against linear
      // costs near 40 on aircondSetup at rho 0.003), and NumericFocus usually
      // gets them through.  Runs before the INTERRUPTED check so an abort
      // during a retry is handled like any other.
      int origFocus = model->get(GRB_IntParam_NumericFocus);
      for (int focus : {2, 3}) {
        std::cerr << "Warning: Gurobi returned NUMERIC for '" << scenarioName
                  << "'; retrying with NumericFocus=" << focus << "\n";
        model->set(GRB_IntParam_NumericFocus, focus);
        model->optimize();
        status = model->get(GRB_IntAttr_Status);
        if (status != GRB_NUMERIC) {
          std::cerr << "  Retry with NumericFocus=" << focus << " ended with status "
                    << status << " for '" << scenarioName << "'\n";
          break;
        }
      }
      model->set(GRB_IntParam_NumericFocus, origFocus);
    }
    if (status == GRB_INTERRUPTED) {
      // interruptSolve() (or Ctrl-C) stopped the solve; the caller must
      // check wasInterrupted() and discard the result without extraction.
      interrupted.store(true, std::memory_order_relaxed);
      return objAndBnd{0.0, 0.0};
    }
    if (status == GRB_OPTIMAL) {
      obj = model->get(GRB_DoubleAttr_ObjVal);
      bound = model->get(GRB_DoubleAttr_ObjBound);
    } else if (status == GRB_TIME_LIMIT &&
      model->get(GRB_IntAttr_SolCount) > 0) {
      obj = model->get(GRB_DoubleAttr_ObjVal);
      bound = model->get(GRB_DoubleAttr_ObjBound);
    } else if (status == GRB_SUBOPTIMAL &&
      model->get(GRB_IntAttr_SolCount) > 0) {
      // Retry cascade for SUBOPTIMAL: try presolve off, then aggressive scaling
      struct RetryConfig {
        const char* name;
        int presolve;   // -1 means don't change
        int scaleFlag;  // -1 means don't change
      };
      static const RetryConfig retries[] = {
        {"presolve off", 0, -1},
        {"scale=2",     -1,  2},
      };
      int origPresolve = model->get(GRB_IntParam_Presolve);
      int origScale = model->get(GRB_IntParam_ScaleFlag);

      for (const auto& retry : retries) {
        std::cerr << "Warning: Gurobi returned SUBOPTIMAL for '"
                  << scenarioName << "'; retrying with " << retry.name << "\n";
        if (retry.presolve >= 0) model->set(GRB_IntParam_Presolve, retry.presolve);
        if (retry.scaleFlag >= 0) model->set(GRB_IntParam_ScaleFlag, retry.scaleFlag);
        model->optimize();
        status = model->get(GRB_IntAttr_Status);
        // restore original settings
        model->set(GRB_IntParam_Presolve, origPresolve);
        model->set(GRB_IntParam_ScaleFlag, origScale);
        if (status == GRB_OPTIMAL) {
          std::cerr << "  Retry '" << retry.name << "' succeeded for '"
                    << scenarioName << "'\n";
          break;
        }
      }

      obj = model->get(GRB_DoubleAttr_ObjVal);
      bound = model->get(GRB_DoubleAttr_ObjBound);
      if (status != GRB_OPTIMAL) {
        // All retries failed; log diagnostics and use best available
        double maxVio = 0.0;
        try { maxVio = model->get(GRB_DoubleAttr_MaxVio); } catch (...) {}
        numericalBailout.store(true, std::memory_order_relaxed);
        std::cerr << "Warning: all retries failed for '" << scenarioName
                  << "'; using best available solution"
                  << " (obj=" << obj << ", bound=" << bound
                  << ", maxVio=" << maxVio << ")\n";
        // static int suboptimalCount = 0;
        // std::string baseName = std::filesystem::path(scenarioName).filename().string();
        // std::string lpFile = "suboptimal_" + baseName + "_" + std::to_string(suboptimalCount++) + ".lp";
        // try {
        //   model->write(lpFile);
        //   std::cerr << "  Model written to " << lpFile << "\n";
        // } catch (...) {
        //   std::cerr << "  (failed to write " << lpFile << ")\n";
        // }
      }
    } else if (status == GRB_NUMERIC) {              // 12
      std::cerr << "Gurobi stopped with NUMERIC difficulties\n";
      model->write("numeric_debug.lp");              // for post-mortem
      throw std::runtime_error("Numerical troubles; wrote model to numeric_debug.lp");
    } else if (status == GRB_CUTOFF) {
      if (tolerateInfeasible_) {
        // No solution better than cutoff found (FW: no improving direction)
        infeasibleDueToCutoff_ = true;
        return {0.0, 0.0};
      }
      throw std::runtime_error("Model cutoff (no solution better than cutoff)");
    } else if (status == GRB_INFEASIBLE) {
      if (tolerateInfeasible_) {
        infeasibleDueToCutoff_ = true;
        return {0.0, 0.0};
      }
      model->computeIIS();
      model->write("model.ilp");             // IIS written in ILP format
      throw std::runtime_error("Model infeasible (IIS written to model.ilp)");
    } else if (status == GRB_UNBOUNDED) {
      throw std::runtime_error("Model unbounded");
    } else if (status == GRB_INF_OR_UNBD) {
      if (tolerateInfeasible_) {
        infeasibleDueToCutoff_ = true;
        return {0.0, 0.0};
      }
      throw std::runtime_error("Model infeasible or unbounded (set InfUnbdInfo=1 to distinguish)");      
    } else {
      throw std::runtime_error("Optimization was stopped with status = " + std::to_string(status));
    }
  }
  catch (const GRBException& e) {
    std::cerr << "Gurobi exception " << e.getErrorCode()
              << ": " << e.getMessage() << "\n";
    throw; // keep the original exception type
  }
  catch (const std::exception& e) {
    std::cerr << "std::exception: " << e.what() << "\n";
    throw; // preserves message instead of "unexpected error"
  }
  catch (...) {
    std::cerr << "Unknown non-std exception\n";
    throw;
  }
  return objAndBnd{obj, bound};
}

// Called from the control thread to stop a solve in flight (best effort).
// GRBModel::terminate is documented as safe to call asynchronously; the
// solveInProgress guard keeps a request that lands between solves from
// leaking into a later solve, since Gurobi latches it.
void GurobiInterface::interruptSolve() {
  if (solveInProgress.load(std::memory_order_relaxed) && model) {
    try {
      model->terminate();
    } catch (...) {
      // best effort only
    }
  }
}

void GurobiInterface::setGurobiParam(const GRB_IntParam param, const std::any &value) {
  try {
    // Attempt to extract as int. If it fails, try a broader type.
    int finalValue;
    if (value.type() == typeid(int) || value.type() == typeid(const int)) {
      finalValue = std::any_cast<int>(value);
    } else if (value.type() == typeid(long)) {
      finalValue = static_cast<int>(std::any_cast<long>(value));
    } else {
      // Fallback: If it's not working, print what the type actually IS
      std::cerr << "Gurobi Param Error: Expected int, got " << value.type().name() << std::endl;
      return;
    }

    model->set(param, finalValue);
    
  } catch (const std::bad_any_cast& e) {
    std::cerr << "Bad any_cast in setGurobiParam: " << e.what() << std::endl;
  } catch (GRBException &e) {
    std::cerr << "Gurobi error: " << e.getMessage() << std::endl;
  }
}
  
void GurobiInterface::setGurobiParam(const GRB_DoubleParam param, const std::any &value) {
  try {
    if (value.type() == typeid(double)) {
      model->set(param, std::any_cast<double>(value));
    } else {
      throw std::runtime_error("double parameter valued needed");
    }
  } catch (const std::bad_any_cast& e) {
    std::cerr << "Type mismatch in parameter value: " << e.what() << std::endl;
  } catch (GRBException &e) {
    std::cerr << "Gurobi error code = " << e.getErrorCode() << std::endl;
    std::cerr << e.getMessage() << std::endl;
  } catch (...) {
    std::cerr << "setGurobiParam: unknown exception" << std::endl;
  }
}


void GurobiInterface::printSolution() {
  // This is for debugging, so we get local copies from the model
  std::cout << "Subproblem name=" << scenarioName << std::endl;
  // Retrieve the objective value.
  double objVal = model->get(GRB_DoubleAttr_ObjVal);
  std::cout << "Optimal objective: " << objVal << std::endl;
  // loop over variables to print their values.
  int lnumVars = model->get(GRB_IntAttr_NumVars);
  GRBVar* lvars = model->getVars();
  for (int i = 0; i < lnumVars; i++) {
    std::cout << lvars[i].get(GRB_StringAttr_VarName)
              << " = " << lvars[i].get(GRB_DoubleAttr_X) << std::endl;
  }
  delete[] lvars; // Free the memory allocated by getVars()

}

std::vector<double> GurobiInterface::copyNonantSolution() const {
  // copy the nonant values from the gurobi object to a dense vector
  // This might never be used...
  std::vector<double> returnVector;
  for (const auto& [nd, indexList] : nonantIndexes) {
    for (const auto& i : indexList) {
      returnVector.push_back(vars[i].get(GRB_DoubleAttr_X));
    }
  }
  return returnVector;
}

void GurobiInterface::extractNonantSolution(std::vector<double>& nonantSolution) const {
  if ( nonantSolution.size() != totalNonantCount ) {
    std::cerr << "nonantSolution.size()=" << nonantSolution.size() <<
      " totalNonantCount=" << totalNonantCount << std::endl;
    throw std::runtime_error("bad call to extractNonantSolution");
  }
  int j = 0;  // good, old-fashion c density...
  for (const auto& [nd, indexList] : nonantIndexes) {
    for (const auto& i : indexList) {
      nonantSolution[j++] = vars[i].get(GRB_DoubleAttr_X);
    }
  }
}

void GurobiInterface::extractNodeSolution(const std::string nodeName,
                                          std::vector<double>& nodeSolution) const {
  auto nI =  nonantIndexes.at(nodeName);  // typing aid
  if ( nodeSolution.size() != nI.size() ) {
    std::cerr << "nodeSolution.size()=" << nodeSolution.size()
              << " nonantIndex[nodeName].size()=" << nI.size()
              << std::endl;
    throw std::runtime_error("bad call to extractNodeSolution");
  }
  int j = 0;
  for (const auto& i : nI) {
    nodeSolution[j++] = vars[i].get(GRB_DoubleAttr_X);
  }
}


std::vector<double> GurobiInterface::copyEntireSolution() const {
  // create a vector and copy the values from the gurobi object to it
  std::vector<double> returnVector(numVars);
  for (int i = 0; i < numVars; ++i) {
    returnVector[i] = vars[i].get(GRB_DoubleAttr_X);
  }
  return returnVector;
}


void GurobiInterface::nonantSolutionVectorToCsv(std::string fileName) const {

  std::ofstream outFile(fileName);
  if (outFile.is_open()) {
    // no header
    for (const auto& [name, i] : nonantNameToIndex) {
      outFile << name << "," << vars[i].get(GRB_DoubleAttr_X) << "\n";
    }
    outFile.close();
  } else {
    throw std::runtime_error("Error trying to write file="+fileName);
  }  
}

bool GurobiInterface::setStartAtCurrent() {
  if (model->get(GRB_IntAttr_SolCount) > 0) {
    int lnumVars = model->get(GRB_IntAttr_NumVars);
    GRBVar* lvars = model->getVars();     
    for (int i = 0; i < lnumVars; ++i) {
      double val = lvars[i].get(GRB_DoubleAttr_X);     // incumbent value
      lvars[i].set(GRB_DoubleAttr_Start, val);         // set as MIP start
    }
    delete[] lvars;
    return true;
  } else {
    return false;
  }
}


void GurobiInterface::clearModel() {
  try {
    // Remove all constraints.
    int numConstrs = model->get(GRB_IntAttr_NumConstrs);
    if (numConstrs > 0) {
       GRBConstr* constrs = model->getConstrs();
       for (int i = 0; i < numConstrs; i++) {
         // Remove each constraint individually; bummer.
         model->remove(constrs[i]);
       }
       delete[] constrs;  // local
    }
    
    // Remove all variables.
    int lnumVars = model->get(GRB_IntAttr_NumVars);
    if (lnumVars > 0) {
       GRBVar* lvars = model->getVars();
       for (int i = 0; i < lnumVars; i++) {
         // Remove each variable individually; bummer.
         model->remove(lvars[i]);
       }
       model->update();
       delete[] lvars;  // local
    }
    delete[] vars;  // this is different than cplex
    vars = nullptr;
  }
  catch (GRBException &e) {
    std::cerr << "Error clearing model; Gurobi error code = " 
              << e.getErrorCode() << ": " << e.getMessage() << std::endl;
    throw std::runtime_error("Error in clearModel");
  }
  numVars = 0;
}


#include <sstream>
#include <cmath>
#include "gurobi_c++.h"

std::string gurobiExprToString(GRBQuadExpr expr) {
    std::ostringstream out;

    // ---- Linear part ----
    GRBLinExpr lin = expr.getLinExpr();

    double constant = lin.getConstant();
    bool first = true;

    if (constant != 0.0) {
        out << constant;
        first = false;
    }

    for (int i = 0; i < lin.size(); ++i) {
        double coef = lin.getCoeff(i);
        GRBVar v = lin.getVar(i);

        if (coef == 0.0) continue;

        if (!first) {
            out << (coef >= 0.0 ? " + " : " - ");
        } else if (coef < 0.0) {
            out << "-";
        }

        double abscoef = std::abs(coef);
        if (abscoef != 1.0) {
            out << abscoef << " * ";
        }

        out << v.get(GRB_StringAttr_VarName);
        first = false;
    }

    // ---- Quadratic part ----
    unsigned int qsize = expr.size();   // number of quadratic terms

    for (unsigned int i = 0; i < qsize; ++i) {
        double coef = expr.getCoeff(i);
        GRBVar v1 = expr.getVar1(i);
        GRBVar v2 = expr.getVar2(i);

        if (coef == 0.0) continue;

        if (!first) {
            out << (coef >= 0.0 ? " + " : " - ");
        } else if (coef < 0.0) {
            out << "-";
        }

        double abscoef = std::abs(coef);
        if (abscoef != 1.0) {
            out << abscoef << " * ";
        }

        out << v1.get(GRB_StringAttr_VarName)
            << " * "
            << v2.get(GRB_StringAttr_VarName);
        first = false;
    }

    if (first) {
        // nothing printed at all
        out << "0";
    }

    return out.str();
}
