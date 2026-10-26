///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "cplex_interface.hpp"
#include <ilcplex/ilocplex.h>
#include <string>
#include <iostream>
#include <exception>
#include <fstream>
#include <unordered_map>

// One-arg constructor delegates to two-arg with empty logFileName
CPLEXInterface::CPLEXInterface(const std::string& scenarioName)
  : CPLEXInterface(scenarioName, std::string())
{ }

// Two-arg constructor. Initializes Concert objects and optionally opens logfile.
CPLEXInterface::CPLEXInterface(const std::string& scenarioName,
                               const std::string& logFileName)
  : SolverInterface(scenarioName),
    env(),
    model(env),
    cplex(model),    // bind CPLEX to the model
    vars(env),
    constraints(env),
    objective(env),
    logStream_(nullptr)
{
  // Make base know about the log file name
  SolverInterface::setLogFile(logFileName);
  
  try {
    // Import model (mpsFileName is expected to be set by the base class)
    cplex.importModel(model, mpsFileName.c_str(), objective, vars, constraints);
    if (vars.getSize() == 0) {
      throw std::runtime_error("No variables in the model for " + mpsFileName);
    }
    cplex.extract(model);
    aborter_ = IloCplex::Aborter(env);
    cplex.use(aborter_);
  } catch (IloException& e) {
    std::cerr << "CPLEX Exception during importModel: " << e.getMessage() << std::endl;
    throw;
  } catch (...) {
    std::cerr << "Unknown Exception occurred during importModel." << std::endl;
    throw;
  }

  // Set up persistent logging (or silence)
  if (!logFileName.empty()) {
    logStream_ = std::make_unique<std::ofstream>(logFileName, std::ios::out | std::ios::app);
    if (!logStream_->is_open()) {
      std::cerr << "Warning: could not open CPLEX log file '" << logFileName << "', using null stream.\n";
      logStream_.reset();
      cplex.setOut(env.getNullStream());
      cplex.setWarning(env.getNullStream());
    } else {
      // Route both normal output and warnings to the persistent stream.
      cplex.setOut(*logStream_);
      cplex.setWarning(*logStream_);
    }
  } else {
    // Default: silence CPLEX textual output
    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
  }

  // create the vector of original costs (one entry per variable, aligned with vars)
  fullC.assign(static_cast<size_t>(vars.getSize()), 0.0);

  try {
    // Map variable name -> index in vars[]
    std::unordered_map<std::string, IloInt> nameToIndex;
    nameToIndex.reserve(static_cast<size_t>(vars.getSize()));

    for (IloInt j = 0; j < vars.getSize(); ++j) {
      nameToIndex.emplace(std::string(vars[j].getName()), j);
    }

    // Fill coefficients for vars that appear in the objective (others remain 0.0)
    IloExpr objExpr = objective.getExpr();
    for (IloExpr::LinearIterator it = objExpr.getLinearIterator(); it.ok(); ++it) {
      IloNumVar v = it.getVar();
      IloNum coef = it.getCoef();
      const std::string vname(v.getName());

      auto pos = nameToIndex.find(vname);
      if (pos != nameToIndex.end()) {
        fullC[static_cast<size_t>(pos->second)] = coef;
      } else {
        std::cerr << "WARNING: objective term variable not found in vars[]: "
                  << vname << std::endl;
      }
    }
    objExpr.end();

    // Now deal with quadratic terms (note: we could avoid the double end and getExpr)
    fullQDiag.assign(static_cast<size_t>(vars.getSize()), 0.0);

    objExpr = objective.getExpr();

    // quadratic
    for (IloExpr::QuadIterator it = objExpr.getQuadIterator(); it.ok(); ++it) {
      IloNumVar v1 = it.getVar1();
      IloNumVar v2 = it.getVar2();
      IloNum coef = it.getCoef();

      std::string n1(v1.getName());
      std::string n2(v2.getName());

      auto i1 = nameToIndex.find(n1);
      auto i2 = nameToIndex.find(n2);
      if (i1 == nameToIndex.end() || i2 == nameToIndex.end()) {
        throw std::runtime_error("Quadratic objective var not found in vars[]");
      }

      int idx1 = static_cast<int>(i1->second);
      int idx2 = static_cast<int>(i2->second);

      if (idx1 == idx2) {
        fullQDiag[static_cast<size_t>(idx1)] += coef;
        hasOrigQuad = true;
      } else {
        hasOrigQuad = true;
        hasOrigQuadCross = true; // not supported yet
      }
    }

    objExpr.end();

    if (hasOrigQuadCross) {
      throw std::runtime_error("Original objective has cross quadratic terms x_i*x_j (i!=j), not supported yet.");
    }
    

  } catch (IloException& e) {
    std::cerr << "CPLEX Exception while reading objective coefficients: "
              << e.getMessage() << std::endl;
  }
  // --- ensure nonant maps are constructed like Gurobi does --------------
  // Build maps that connect solver variable indexes to nonant names/indices.
  _buildNonantsMaps();  
}

// Destructor: end CPLEX objects then close logfile and end env
CPLEXInterface::~CPLEXInterface() {
  try {
    // End Concert objects in safe order
    try { cplex.end(); } catch(...) {}
    try { objective.end(); } catch(...) {}
    try { constraints.end(); } catch(...) {}
    try { vars.end(); } catch(...) {}
    try { model.end(); } catch(...) {}

    // Close logfile stream (if any)
    if (logStream_ && logStream_->is_open()) {
      logStream_->flush();
      logStream_->close();
      logStream_.reset();
    }

    // Finally end the environment
    try { env.end(); } catch(...) {}
  } catch (...) {
    // Destructors must not throw
  }
}

void CPLEXInterface::fullObjectiveRebuild() {
  if (WOn && !WPtr) {
    throw std::runtime_error("CPLEX fullObjectiveRebuild: WOn=true but WPtr is null");
  }
  
  if (proxOn && !zPtr) {
    throw std::runtime_error("CPLEX fullObjectiveRebuild: proxOn=true but zPtr is null");
  }
  if (rho.size() != static_cast<size_t>(totalNonantCount)) {
    throw std::runtime_error("CPLEX fullObjectiveRebuild: rho size mismatch vs totalNonantCount");
  }
  if (WPtr && WPtr->size() != static_cast<size_t>(totalNonantCount)) {
    throw std::runtime_error("CPLEX fullObjectiveRebuild: W size mismatch vs totalNonantCount");
  }
  if (zPtr && zPtr->size() != static_cast<size_t>(totalNonantCount)) {
    throw std::runtime_error("CPLEX fullObjectiveRebuild: z size mismatch vs totalNonantCount");
  }
  for (int vidx : nonantDenseIndexes) {
    if (vidx < 0 || vidx >= vars.getSize()) {
      throw std::runtime_error("CPLEX fullObjectiveRebuild: nonant index out of range: " + std::to_string(vidx));
    }
  }

  IloExpr expr(env);
  try {
    // baseline linear
    for (IloInt j = 0; j < vars.getSize(); ++j) {
      IloNum c = fullC[static_cast<size_t>(j)];
      if (c != 0.0) expr += c * vars[j];
    }
    // baseline diagonal quad
    if (!fullQDiag.empty()) {
      for (IloInt j = 0; j < vars.getSize(); ++j) {
        IloNum q = fullQDiag[static_cast<size_t>(j)];
        if (q != 0.0) expr += q * vars[j] * vars[j];
      }
    }

    // nonant deltas and prox quads
    int pos = 0;
    for (int vidx : nonantDenseIndexes) {
      IloNum delta = 0.0;
      if (WOn)    delta += (*WPtr)[pos];
      if (proxOn) delta -= rho[pos] * (*zPtr)[pos];

      if (delta != 0.0) expr += delta * vars[vidx];
      if (proxOn) expr += (rho[pos] / 2.0) * vars[vidx] * vars[vidx];
      ++pos;
    }

    objective.setExpr(expr);
    expr.end();

    rhoChanged = false;

  } catch (IloException& e) {
    try { expr.end(); } catch(...) {}
    std::cerr << "CPLEX IloException in fullObjectiveRebuild: " << e.getMessage() << std::endl;
    numericalBailout.store(true, std::memory_order_relaxed);
    return;
  } catch (...) {
    try { expr.end(); } catch(...) {}
    std::cerr << "Unknown exception in CPLEX fullObjectiveRebuild()" << std::endl;
    numericalBailout.store(true, std::memory_order_relaxed);
    return;
  }
}


void CPLEXInterface::applyBoundsToSolver() {
  for (int pos = 0; pos < totalNonantCount; ++pos) {
    int vidx = nonantDenseIndexes[pos];
    vars[vidx].setLB(nonantLowerBounds[pos]);
    vars[vidx].setUB(nonantUpperBounds[pos]);
  }
}

double CPLEXInterface::getVarLB(int varIdx) const {
  return vars[varIdx].getLB();
}

double CPLEXInterface::getVarUB(int varIdx) const {
  return vars[varIdx].getUB();
}


void CPLEXInterface::_buildNonantsMaps() {
  std::unordered_set<std::string> nonantSet;
  std::unordered_set<std::string> modelVarSet;

  // build nonantSet and nodeNonantCounts
  for (const auto& [ndn, nd] : treeNodeDataMap) {
    const auto& currentList = nd.nonAnts;
    nonantSet.insert(currentList.begin(), currentList.end());
    nodeNonantCounts[ndn] = currentList.size();
  }

  // map var names to indices and collect model vars that are nonants
  for (IloInt i = 0; i < vars.getSize(); ++i) {
    const std::string vname = vars[i].getName();
    varNameToIndex[vname] = static_cast<int>(i);
    if (nonantSet.count(vname)) {
      modelVarSet.insert(vname);
      nonantNameToIndex[vname] = static_cast<int>(i);
      nonantIndexToName[static_cast<int>(i)] = vname;
    }
  }

  // check for missing nonants in the model (do this BEFORE building indexes)
  std::unordered_set<std::string> difference;
  for (const auto& vname : nonantSet) {
    if (modelVarSet.find(vname) == modelVarSet.end()) {
      difference.insert(vname);
    }
  }
  if (!difference.empty()) {
    std::cerr << "The following variables were in the nonant json, but not the model\n";
    for (const auto& vname : difference) std::cerr << vname << "\n";
    throw std::runtime_error("Nonant json vs model var name mismatch: " +
                             nonantFileName + " " + mpsFileName);
  }

  // build list of indexes for each tree node (no operator[] on map!)
  for (const auto& [ndn, nd] : treeNodeDataMap) {
    const auto& currentList = nd.nonAnts;
    for (const auto& vname : currentList) {
      auto it = nonantNameToIndex.find(vname);
      if (it == nonantNameToIndex.end()) {
        // Should not happen because we already checked 'difference', but keep it loud.
        throw std::runtime_error("Nonant not in model after difference-check: " + vname);
      }
      nonantIndexes[ndn].push_back(it->second);
    }
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
  // sanity: fullC must align with vars
  if (fullC.size() != static_cast<size_t>(vars.getSize())) {
    std::cerr << "fullC.size()=" << fullC.size()
              << " vars.getSize()=" << vars.getSize() << "\n";
    throw std::runtime_error("fullC not aligned with vars (constructor should fill fullC)");
  }

  // Optional sanity if you expect dense count equals totalNonantCount:
  if (totalNonantCount != 0 &&
      nonantDenseIndexes.size() != static_cast<size_t>(totalNonantCount)) {
    std::cerr << "WARNING: nonantDenseIndexes.size()=" << nonantDenseIndexes.size()
              << " totalNonantCount=" << totalNonantCount << "\n";
  }

  std::unordered_set<int> seen;
  for (int vidx : nonantDenseIndexes) {
    if (!seen.insert(vidx).second) {
      throw std::runtime_error("Duplicate variable index in nonantDenseIndexes: " + std::to_string(vidx));
    }
  }
  
}


objAndBnd CPLEXInterface::solve() {
  double obj = 0.0;
  double bound = 0.0;

  interrupted.store(false, std::memory_order_relaxed);
  if (abortRequested()) {
    interrupted.store(true, std::memory_order_relaxed);
    return objAndBnd{0.0, 0.0};
  }
  if (aborter_.getImpl()) {
    aborter_.clear();  // reset any abort request from a previous solve
  }

  // Ensure CPLEX output is set to the persistent stream or to null stream
  if (logStream_ && logStream_->is_open()) {
    cplex.setOut(*logStream_);
    cplex.setWarning(*logStream_);
  } else {
    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
  }

  try {
    // Solve the problem
    bool solved = cplex.solve();
    auto st = cplex.getStatus();
    if (cplex.getCplexStatus() == IloCplex::AbortUser) {
      // interruptSolve() stopped the solve (possibly with an incumbent);
      // the caller must check wasInterrupted() and discard the result.
      interrupted.store(true, std::memory_order_relaxed);
      return objAndBnd{0.0, 0.0};
    }
    if (solved) {
      if (st != IloAlgorithm::Optimal && st != IloAlgorithm::Feasible) {
        std::cerr << "Warning: CPLEX solve() returned true with status "
                  << cplex.getStatus() << "; using available solution\n";
      }
      obj = cplex.getObjValue();
      bound = obj;
      if (cplex.isMIP()) {
        bound = cplex.getBestObjValue();
      }
    } else {
      throw std::runtime_error("CPLEX solve failed with status: "
                               + std::to_string(static_cast<int>(st)));
    }
  } catch (const IloException& e) {
    std::cerr << "CPLEX Exception in solve(): " << e.getMessage() << "\n";
    throw std::runtime_error(std::string("CPLEX solve IloException: ") + e.getMessage());
  } catch (...) {
    std::cerr << "Unknown Exception occurred in solve().\n";
    throw;
  }

  // Do NOT close the persistent log stream here; destructor will handle it.
  return objAndBnd{obj, bound};
}

// Called from the control thread to stop a solve in flight (best effort).
// IloCplex::Aborter is designed for asynchronous aborts.
void CPLEXInterface::interruptSolve() {
  try {
    if (aborter_.getImpl()) {
      aborter_.abort();
    }
  } catch (...) {
    // best effort only
  }
}

void CPLEXInterface::printSolution() {
  std::cout << "Subproblem name=" << scenarioName << std::endl;
  try {
    std::cout << "Objective value: " << cplex.getObjValue() << std::endl;
    std::cout << "Variable values: " << std::endl;
    IloNumArray vals(env);
    cplex.getValues(vals, vars);
    for (IloInt i = 0; i < vals.getSize(); ++i) {
      std::cout << vars[i].getName() << ": " << vals[i] << std::endl;
    }
  } catch (IloException& e) {
    std::cerr << "CPLEX Exception: " << e.getMessage() << std::endl;
  } catch (...) {
    std::cerr << "Unknown Exception occurred." << std::endl;
  }
}

std::vector<double> CPLEXInterface::copyNonantSolution() const {
  std::vector<double> returnVector;
  IloNumArray vals(env);
  cplex.getValues(vals, vars);
  for (const auto& [nd, indexList] : nonantIndexes) {
    for (const auto& i : indexList) {
      returnVector.push_back(vals[i]);
    }
  }
  return returnVector;
}

std::vector<double> CPLEXInterface::copyEntireSolution() const {
  IloNumArray vals(env);
  cplex.getValues(vals, vars);
  std::vector<double> returnVector(vals.getSize());
  for (IloInt i = 0; i < vals.getSize(); ++i) {
    returnVector[i] = vals[i];
  }
  return returnVector;
}

void CPLEXInterface::extractNonantSolution(std::vector<double>& nonantSolution) const {
  if ( nonantSolution.size() != totalNonantCount ) {
    std::cerr << "nonantSolution.size()=" << nonantSolution.size()
              << " totalNonantCount=" << totalNonantCount << std::endl;
    throw std::runtime_error("bad call to extractNonantSolution");
  }
  IloNumArray vals(env);
  cplex.getValues(vals, vars);
  int j = 0;
  for (const auto& [nd, indexList] : nonantIndexes) {
    for (const auto& i : indexList) {
      nonantSolution[j++] = vals[i];
    }
  }
}

void CPLEXInterface::extractNodeSolution(const std::string nodeName,
                                          std::vector<double>& nodeSolution) const {
  auto nI =  nonantIndexes.at(nodeName);
  if ( nodeSolution.size() != nI.size() ) {
    std::cerr << "nodeSolution.size()=" << nodeSolution.size()
              << " nonantIndex[nodeName].size()=" << nI.size()
              << std::endl;
    throw std::runtime_error("bad call to extractNodeSolution");
  }
  IloNumArray vals(env);
  cplex.getValues(vals, vars);
  int j = 0;
  for (const auto& i : nI) {
    nodeSolution[j++] = vals[i];
  }
}

void CPLEXInterface::nonantSolutionVectorToCsv(std::string fileName) const {
  IloNumArray vals(env);
  cplex.getValues(vals, vars);

  std::ofstream outFile(fileName);
  if (outFile.is_open()) {
    for (const auto& [name, i] : nonantNameToIndex) {
      outFile << name << "," << vals[i] << "\n";
    }
    outFile.close();
  } else {
    throw std::runtime_error("Error try to write file="+fileName);
  }
}

bool CPLEXInterface::setStartAtCurrent() {
  auto st = cplex.getStatus();
  if (!(st == IloAlgorithm::Optimal || st == IloAlgorithm::Feasible)) {
    return false;
  }
  IloNumArray vals(env);
  try {
    cplex.getValues(vals, vars);
    cplex.addMIPStart(vars, vals, IloCplex::MIPStartAuto, "fromCurrent");
    return true;
  } catch (const IloException& e) {
    return false;
  }
}

void CPLEXInterface::clearModel() {
  // End old objects (order matters)
  try { cplex.end(); } catch(...) {}
  try { objective.end(); } catch(...) {}
  try { constraints.end(); } catch(...) {}
  try { vars.end(); } catch(...) {}
  try { model.end(); } catch(...) {}

  // Recreate
  model = IloModel(env);
  vars = IloNumVarArray(env);
  constraints = IloRangeArray(env);
  objective = IloMinimize(env, 0.0);
  model.add(objective);
  cplex = IloCplex(model);

  // Restore logging policy
  if (logStream_ && logStream_->is_open()) {
    cplex.setOut(*logStream_);
    cplex.setWarning(*logStream_);
  } else {
    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
  }
}


// gurobi cannot do this on the fly, but CPLEX can
void CPLEXInterface::setLogFile(const std::string fileName) {
  SolverInterface::setLogFile(fileName);

  // tear down old
  if (logStream_ && logStream_->is_open()) {
    logStream_->flush();
    logStream_->close();
  }
  logStream_.reset();

  if (!logFileName.empty()) {
    logStream_ = std::make_unique<std::ofstream>(logFileName, std::ios::out | std::ios::app);
    if (logStream_->is_open()) {
      cplex.setOut(*logStream_);
      cplex.setWarning(*logStream_);
      return;
    }
    // fallback
    logStream_.reset();
  }
  cplex.setOut(env.getNullStream());
  cplex.setWarning(env.getNullStream());
}
