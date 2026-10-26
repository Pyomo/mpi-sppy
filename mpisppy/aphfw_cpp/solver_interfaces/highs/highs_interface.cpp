///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "highs_interface.hpp"
#include <string>
#include <iostream>
#include <exception>
#include <limits>
#include <algorithm>

std::atomic<int> HiGHSInterface::processThreads_{0};
std::atomic<bool> HiGHSInterface::threadPoolStarted_{false};

HiGHSInterface::HiGHSInterface(const std::string& scenarioName,
                               const std::string& logFileName):
  SolverInterface(scenarioName),
  highs_(std::make_unique<Highs>()),
  cbState_(std::make_unique<CallbackState>())
{
  highsCheck(highs_->setCallback(callback_, cbState_.get()), "setCallback");
  if (logFileName != "") {
    cbState_->log.open(logFileName, std::ios::app);
    if (!cbState_->log) {
      throw std::runtime_error("HiGHS: cannot open log file " + logFileName);
    }
    highsCheck(highs_->setOptionValue("log_to_console", false), "log_to_console");
    highsCheck(highs_->startCallback(kCallbackLogging), "startCallback(logging)");
  }
  highsCheck(highs_->startCallback(kCallbackSimplexInterrupt), "startCallback(simplex)");
  highsCheck(highs_->startCallback(kCallbackIpmInterrupt), "startCallback(ipm)");
  highsCheck(highs_->startCallback(kCallbackMipInterrupt), "startCallback(mip)");

  highsCheck(highs_->readModel(mpsFileName), "readModel(" + mpsFileName + ")");

  const HighsLp& lp = highs_->getLp();
  if (lp.sense_ != ObjSense::kMinimize) {
    throw std::runtime_error("HiGHS: " + mpsFileName + " is a maximization; "
                             "the FW-APH solves assume minimization.");
  }
  const HighsHessian& hessian = highs_->getModel().hessian_;
  if (hessian.dim_ > 0 &&
      std::any_of(hessian.value_.begin(), hessian.value_.end(),
                  [](double v) { return v != 0.0; })) {
    hasOrigQuad = true;
    throw std::runtime_error(
      "HiGHS: the objective in " + mpsFileName + " has quadratic terms.  With a "
      "quadratic objective the FW-APH Lagrangian and gradient solves are MIQPs, "
      "which HiGHS cannot solve; build aphfw_cpp for Gurobi, CPLEX or Xpress.");
  }
  // The other backends rebuild the objective without its constant, so this
  // one drops it too and the objective values agree.
  highsCheck(highs_->changeObjectiveOffset(0.0), "changeObjectiveOffset");

  numVars = lp.num_col_;
  if (static_cast<int>(lp.col_names_.size()) != numVars) {
    throw std::runtime_error("HiGHS: " + mpsFileName + " has no column names");
  }
  fullC = lp.col_cost_;
  fullQDiag.assign(static_cast<size_t>(numVars), 0.0);

  // the parent class loads the nonants from json
  _buildNonantsMaps();
}

void HiGHSInterface::callback_(int callbackType, const std::string& message,
                               const HighsCallbackOutput* /*dataOut*/,
                               HighsCallbackInput* dataIn, void* userData) {
  auto* state = static_cast<CallbackState*>(userData);
  if (callbackType == kCallbackLogging) {
    std::lock_guard<std::mutex> lock(state->logMutex);
    state->log << message;
    return;
  }
  if (state->stopRequested.load(std::memory_order_relaxed)) {
    dataIn->user_interrupt = true;
  }
}

void HiGHSInterface::highsCheck(HighsStatus status, const std::string& what) const {
  if (status == HighsStatus::kError) {
    throw std::runtime_error("HiGHS error in " + what + " for '" + scenarioName + "'");
  }
}

bool HiGHSInterface::hasIntegerCols() const {
  const auto& integrality = highs_->getLp().integrality_;
  return std::any_of(integrality.begin(), integrality.end(),
                     [](HighsVarType t) { return t != HighsVarType::kContinuous; });
}

void HiGHSInterface::fullObjectiveRebuild() {
  try {
    if (WOn && !WPtr) {
      throw std::runtime_error("HiGHS fullObjectiveRebuild: WOn=true but WPtr is null");
    }
    if (proxOn && !zPtr) {
      throw std::runtime_error("HiGHS fullObjectiveRebuild: proxOn=true but zPtr is null");
    }
    if (rho.size() != static_cast<size_t>(totalNonantCount)) {
      throw std::runtime_error("HiGHS fullObjectiveRebuild: rho size mismatch vs totalNonantCount");
    }
    if (WPtr && WPtr->size() != static_cast<size_t>(totalNonantCount)) {
      throw std::runtime_error("HiGHS fullObjectiveRebuild: W size mismatch vs totalNonantCount");
    }
    if (zPtr && zPtr->size() != static_cast<size_t>(totalNonantCount)) {
      throw std::runtime_error("HiGHS fullObjectiveRebuild: z size mismatch vs totalNonantCount");
    }

    // baseline linear, then the nonant deltas
    std::vector<double> cost(fullC);
    std::vector<double> qDiag(static_cast<size_t>(numVars), 0.0);
    int pos = 0;
    for (int vidx : nonantDenseIndexes) {
      if (vidx < 0 || vidx >= numVars) {
        throw std::runtime_error("HiGHS fullObjectiveRebuild: nonant index out of range: " + std::to_string(vidx));
      }
      if (WOn)    cost[vidx] += (*WPtr)[pos];
      if (proxOn) {
        cost[vidx] -= rho[pos] * (*zPtr)[pos];
        qDiag[vidx] = rho[pos];  // (1/2) x'Qx convention
      }
      ++pos;
    }
    if (numVars > 0) {
      highsCheck(highs_->changeColsCost(0, numVars - 1, cost.data()), "changeColsCost");
    }
    highsCheck(highs_->changeObjectiveSense(ObjSense::kMinimize), "changeObjectiveSense");

    // Diagonal Hessian over all current columns (HiGHSNL has g columns past
    // numVars, which get no entries).  An empty one makes the model linear.
    HighsHessian hessian;
    if (proxOn) {
      const HighsInt numCol = highs_->getNumCol();
      hessian.dim_ = numCol;
      hessian.format_ = HessianFormat::kTriangular;
      hessian.start_.assign(static_cast<size_t>(numCol) + 1, 0);
      for (HighsInt j = 0; j < numCol; ++j) {
        if (j < numVars && qDiag[j] != 0.0) {
          hessian.index_.push_back(j);
          hessian.value_.push_back(qDiag[j]);
        }
        hessian.start_[j + 1] = static_cast<HighsInt>(hessian.index_.size());
      }
    }
    if (proxOn || highs_->getModel().hessian_.dim_ > 0) {
      highsCheck(highs_->passHessian(hessian), "passHessian");
    }

    rhoChanged = false;

  } catch (const std::exception &e) {
    std::cerr << "std::exception in HiGHS fullObjectiveRebuild: " << e.what() << std::endl;
    numericalBailout.store(true, std::memory_order_relaxed);
    return;
  } catch (...) {
    std::cerr << "Unknown exception in HiGHS fullObjectiveRebuild()" << std::endl;
    numericalBailout.store(true, std::memory_order_relaxed);
    return;
  }
}


void HiGHSInterface::applyBoundsToSolver() {
  // Touch only the bounds that change, as the Gurobi backend does.
  const HighsLp& lp = highs_->getLp();
  for (int pos = 0; pos < totalNonantCount; ++pos) {
    int vidx = nonantDenseIndexes[pos];
    if (lp.col_lower_[vidx] != nonantLowerBounds[pos] ||
        lp.col_upper_[vidx] != nonantUpperBounds[pos]) {
      highsCheck(highs_->changeColBounds(vidx, nonantLowerBounds[pos],
                                         nonantUpperBounds[pos]),
                 "changeColBounds");
    }
  }
}

double HiGHSInterface::getVarLB(int varIdx) const {
  return highs_->getLp().col_lower_.at(varIdx);
}

double HiGHSInterface::getVarUB(int varIdx) const {
  return highs_->getLp().col_upper_.at(varIdx);
}


void HiGHSInterface::_buildNonantsMaps() {
  // Called by the constructor only (as in the Gurobi backend).
  std::unordered_set<std::string> modelVarSet;   // model var names
  std::unordered_set<std::string> nonantSet;  // nonant var names

  for (const auto& [ndn, nd] : treeNodeDataMap) {
    const auto& currentList = nd.nonAnts;
    nonantSet.insert(currentList.begin(), currentList.end());
    nodeNonantCounts[ndn] = currentList.size();
  }

  const auto& names = highs_->getLp().col_names_;
  for (int i = 0; i < numVars; ++i) {
    const auto& vname = names[i];
    varNameToIndex[vname] = i;
    if (nonantSet.count(vname)) {
      modelVarSet.insert(vname);  // local
      nonantNameToIndex[vname] = i;
      nonantIndexToName[i] = vname;
    }
  }

  for (const auto& [ndn, nd] : treeNodeDataMap) {
    for (const auto& vname : nd.nonAnts) {
      auto it = nonantNameToIndex.find(vname);
      if (it == nonantNameToIndex.end()) {
        throw std::runtime_error("Nonant not in model: " + vname);
      }
      nonantIndexes[ndn].push_back(it->second);
    }
  }
  std::unordered_set<std::string> difference;
  for (auto &vname : nonantSet) {
    if (modelVarSet.find(vname) == modelVarSet.end()) {
      difference.insert(vname);
    }
  }
  if (!difference.empty()) {
    std::cerr << "The following variables were in the nonant json,"
              << " but not the model" << std::endl;
    for (auto &vname : difference) {
      std::cerr << vname << std::endl;
    }
    throw std::runtime_error("nonant vs model var name mismatch for " \
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


objAndBnd HiGHSInterface::solve() {
  constexpr double inf = std::numeric_limits<double>::infinity();

  interrupted.store(false, std::memory_order_relaxed);
  if (abortRequested()) {
    interrupted.store(true, std::memory_order_relaxed);
    return objAndBnd{0.0, 0.0};
  }

  const bool isMIP = hasIntegerCols();
  if (isMIP && highs_->getModel().hessian_.dim_ > 0) {
    throw std::runtime_error("HiGHS: '" + scenarioName + "' would need an MIQP "
                             "solve (quadratic objective with integer variables), "
                             "which HiGHS cannot do.");
  }
  if (startPending_) {
    if (isMIP && solution_.size() == static_cast<size_t>(highs_->getNumCol())) {
      HighsSolution start;
      start.col_value = solution_;
      start.value_valid = true;
      // A start HiGHS cannot use is a warning, not an error; solve cold then.
      highs_->setSolution(start);
    }
    startPending_ = false;
  }

  // interruptSolve() only raises the stop flag while this flag is up; the
  // guard clears both on every exit path.
  struct SolveScope {
    std::atomic<bool>& inProgress;
    std::atomic<bool>& stop;
    ~SolveScope() {
      inProgress.store(false, std::memory_order_relaxed);
      stop.store(false, std::memory_order_relaxed);
    }
  } solveScope{solveInProgress, cbState_->stopRequested};
  if (const int threads = processThreads_.load(); threads > 0) {
    highsCheck(highs_->setOptionValue("threads", static_cast<HighsInt>(threads)),
               "setOptionValue(threads)");
  }
  threadPoolStarted_.store(true);

  cbState_->stopRequested.store(false, std::memory_order_relaxed);
  solveInProgress.store(true, std::memory_order_relaxed);

  if (highs_->run() == HighsStatus::kError) {
    throw std::runtime_error("HiGHS: run() failed for '" + scenarioName +
                             "'; the HiGHS log says why");
  }
  const HighsModelStatus status = highs_->getModelStatus();
  const HighsInfo& info = highs_->getInfo();
  const bool haveSolution = info.primal_solution_status == kSolutionStatusFeasible;

  if (status == HighsModelStatus::kInterrupt) {
    // interruptSolve() stopped the solve; the caller must check
    // wasInterrupted() and discard the result without extraction.
    interrupted.store(true, std::memory_order_relaxed);
    return objAndBnd{0.0, 0.0};
  }

  double obj, bound;
  if (status == HighsModelStatus::kOptimal) {
    obj = info.objective_function_value;
    bound = isMIP ? info.mip_dual_bound : obj;
  } else if ((status == HighsModelStatus::kTimeLimit ||
              status == HighsModelStatus::kIterationLimit) && haveSolution) {
    obj = info.objective_function_value;
    // a continuous solve stopped early has no bound to offer
    bound = isMIP ? info.mip_dual_bound : -inf;
  } else if (status == HighsModelStatus::kInfeasible ||
             status == HighsModelStatus::kUnboundedOrInfeasible) {
    if (tolerateInfeasible_) {
      // No solution better than the cutoff (FW: no improving direction)
      infeasibleDueToCutoff_ = true;
      return {0.0, 0.0};
    }
    throw std::runtime_error("HiGHS: '" + scenarioName + "' is " +
                             highs_->modelStatusToString(status));
  } else if (status == HighsModelStatus::kUnbounded) {
    throw std::runtime_error("HiGHS: '" + scenarioName + "' is unbounded");
  } else if (haveSolution) {
    // The Gurobi backend's SUBOPTIMAL path: use what there is and say so.
    obj = info.objective_function_value;
    bound = isMIP ? info.mip_dual_bound : -inf;
    numericalBailout.store(true, std::memory_order_relaxed);
    std::cerr << "Warning: HiGHS ended '" << scenarioName << "' with status '"
              << highs_->modelStatusToString(status)
              << "'; using its best solution (obj=" << obj << ")\n";
  } else {
    throw std::runtime_error("HiGHS: '" + scenarioName + "' stopped with status '" +
                             highs_->modelStatusToString(status) +
                             "' and no solution");
  }
  solution_ = highs_->getSolution().col_value;
  return objAndBnd{obj, bound};
}

// Called from the control thread to stop a solve in flight (best effort).
// The callback polls the flag; the HiGHS QP solver has no interrupt callback,
// so an NL solve runs to the end.
void HiGHSInterface::interruptSolve() {
  if (solveInProgress.load(std::memory_order_relaxed) && cbState_) {
    cbState_->stopRequested.store(true, std::memory_order_relaxed);
  }
}

void HiGHSInterface::setHiGHSOption(const std::string& name, double value) {
  highsCheck(highs_->setOptionValue(name, value), "setOptionValue(" + name + ")");
}

void HiGHSInterface::requestThreads(int threads) {
  int current = processThreads_.load();
  if (threads <= current) return;
  if (threadPoolStarted_.load()) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
      std::cerr << "Warning: HiGHS is already running on " << current
                << " threads; a request for " << threads << " is ignored\n";
    }
    return;
  }
  while (threads > current && !processThreads_.compare_exchange_weak(current, threads)) {}
}

void HiGHSInterface::exportModel(const std::string fname) {
  highsCheck(highs_->writeModel(fname), "writeModel(" + fname + ")");
}


void HiGHSInterface::printSolution() {
  std::cout << "Subproblem name=" << scenarioName << std::endl;
  std::cout << "Objective: " << highs_->getInfo().objective_function_value << std::endl;
  const auto& names = highs_->getLp().col_names_;
  for (size_t i = 0; i < solution_.size() && i < names.size(); ++i) {
    std::cout << names[i] << " = " << solution_[i] << std::endl;
  }
}

std::vector<double> HiGHSInterface::copyNonantSolution() const {
  std::vector<double> returnVector;
  for (const auto& [nd, indexList] : nonantIndexes) {
    for (const auto& i : indexList) {
      returnVector.push_back(solution_.at(i));
    }
  }
  return returnVector;
}

void HiGHSInterface::extractNonantSolution(std::vector<double>& nonantSolution) const {
  if ( nonantSolution.size() != totalNonantCount ) {
    std::cerr << "nonantSolution.size()=" << nonantSolution.size() <<
      " totalNonantCount=" << totalNonantCount << std::endl;
    throw std::runtime_error("bad call to extractNonantSolution");
  }
  int j = 0;
  for (const auto& [nd, indexList] : nonantIndexes) {
    for (const auto& i : indexList) {
      nonantSolution[j++] = solution_.at(i);
    }
  }
}

void HiGHSInterface::extractNodeSolution(const std::string nodeName,
                                         std::vector<double>& nodeSolution) const {
  const auto& nI = nonantIndexes.at(nodeName);
  if ( nodeSolution.size() != nI.size() ) {
    std::cerr << "nodeSolution.size()=" << nodeSolution.size()
              << " nonantIndex[nodeName].size()=" << nI.size()
              << std::endl;
    throw std::runtime_error("bad call to extractNodeSolution");
  }
  int j = 0;
  for (const auto& i : nI) {
    nodeSolution[j++] = solution_.at(i);
  }
}

std::vector<double> HiGHSInterface::copyEntireSolution() const {
  if (solution_.size() < static_cast<size_t>(numVars)) {
    throw std::runtime_error("HiGHS copyEntireSolution: no solution for '" +
                             scenarioName + "'");
  }
  return std::vector<double>(solution_.begin(), solution_.begin() + numVars);
}


void HiGHSInterface::nonantSolutionVectorToCsv(std::string fileName) const {
  std::ofstream outFile(fileName);
  if (outFile.is_open()) {
    // no header
    for (const auto& [name, i] : nonantNameToIndex) {
      outFile << name << "," << solution_.at(i) << "\n";
    }
    outFile.close();
  } else {
    throw std::runtime_error("Error trying to write file="+fileName);
  }
}

bool HiGHSInterface::setStartAtCurrent() {
  if (solution_.empty()) return false;
  startPending_ = true;
  return true;
}


void HiGHSInterface::clearModel() {
  highsCheck(highs_->clearModel(), "clearModel");
  solution_.clear();
  startPending_ = false;
  numVars = 0;
}
