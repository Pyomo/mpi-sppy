#include "xpress_interface.hpp"
#include <string>
#include <iostream>
#include <sstream>
#include <exception>
#include <cmath>
#include <numeric>
#include <fstream>
#include <cstdlib>

// ---- static members ----
std::once_flag XpressInterface::s_initFlag_;

void XpressInterface::globalInit() {
  // XPRSinit needs the path to the directory containing xpauth.xpr (the license file).
  // Try: XPRESSDIR/bin first (common layout), then XPRESSDIR itself, then NULL (library default).
  const char* xpressdir = std::getenv("XPRESSDIR");

  auto tryInit = [](const char* path) -> int {
    return XPRSinit(path);
  };

  int ret = -1;
  std::string usedPath = "(null)";

  if (xpressdir && *xpressdir) {
    // Try XPRESSDIR/bin (where xpauth.xpr typically lives)
    std::string binPath = std::string(xpressdir) + "/bin";
    ret = tryInit(binPath.c_str());
    usedPath = binPath;

    if (ret != 0) {
      // Try XPRESSDIR itself
      ret = tryInit(xpressdir);
      usedPath = xpressdir;
    }
  }

  if (ret != 0) {
    // Last resort: NULL (library default search)
    ret = tryInit(nullptr);
    usedPath = "(null)";
  }

  if (ret != 0) {
    throw std::runtime_error(
      "XPRSinit failed (code " + std::to_string(ret) + ")."
      "\nEnsure XPRESSDIR environment variable points to your Xpress installation.");
  }

  std::atexit([] { XPRSfree(); });
}

// ---- helper ----
void XpressInterface::xprsCheck(int ret, const char* context) const {
  if (ret != 0) {
    char msg[512] = {0};
    if (prob_) XPRSgetlasterror(prob_, msg);
    throw std::runtime_error(
      std::string("Xpress error in ") + context +
      " (code " + std::to_string(ret) + "): " + msg);
  }
}

// ---- constructor ----
XpressInterface::XpressInterface(const std::string& scenarioName,
                                 const std::string& logFileName)
  : SolverInterface(scenarioName)
{
  // One-time global init
  std::call_once(s_initFlag_, globalInit);

  try {
    // Create problem
    int ret = XPRScreateprob(&prob_);
    if (ret != 0 || !prob_) {
      throw std::runtime_error("XPRScreateprob failed for " + scenarioName);
    }

    // Set up logging
    if (!logFileName.empty()) {
      SolverInterface::setLogFile(logFileName);
      XPRSsetlogfile(prob_, logFileName.c_str());
    } else {
      // Silence output
      XPRSsetintcontrol(prob_, XPRS_OUTPUTLOG, XPRS_OUTPUTLOG_NO_OUTPUT);
    }

    // Read MPS file
    xprsCheck(XPRSreadprob(prob_, mpsFileName.c_str(), ""),
              "XPRSreadprob");

    // Get number of variables
    xprsCheck(XPRSgetintattrib(prob_, XPRS_COLS, &numVars),
              "XPRSgetintattrib(COLS)");

    if (numVars == 0) {
      throw std::runtime_error("No variables in the model for " + mpsFileName);
    }

    // Load variable names
    loadVarNames();

    // Determine if MIP
    int mipents = 0;
    xprsCheck(XPRSgetintattrib(prob_, XPRS_MIPENTS, &mipents),
              "XPRSgetintattrib(MIPENTS)");
    isMIP_ = (mipents > 0);

    // Allocate solution vector
    solution_.assign(numVars, 0.0);

    // Get original linear objective coefficients
    fullC.assign(numVars, 0.0);
    xprsCheck(XPRSgetobj(prob_, fullC.data(), 0, numVars - 1),
              "XPRSgetobj");

    // Build nonant maps (parent loaded nonants from JSON)
    _buildNonantsMaps();

    // Extract original diagonal quadratic terms
    // Xpress convention: objective = c^T x + 0.5 * x^T Q x
    // So Q[i][i] from Xpress gives 0.5 * Q[i][i] * x_i^2 contribution.
    // We store fullQDiag[i] = actual coefficient of x_i^2 = 0.5 * Q[i][i].
    fullQDiag.assign(numVars, 0.0);
    for (int j = 0; j < numVars; ++j) {
      double qval = 0.0;
      // XPRSgetqobj returns the Q matrix element (not halved)
      int qret = XPRSgetqobj(prob_, j, j, &qval);
      if (qret == 0 && qval != 0.0) {
        fullQDiag[j] = qval / 2.0;  // convert to actual coefficient of x_i^2
        hasOrigQuad = true;
      }
    }

    // Check for cross-quadratic terms (off-diagonal)
    // We only support diagonal Q for now, same as Gurobi/CPLEX.
    // A full check would require iterating the Q matrix, which is expensive.
    // We rely on the same convention as the other solvers: if the MPS has
    // cross terms, results will be incorrect but we don't detect it here.
    // (The other solvers detect it via their expression APIs.)

  } catch (const std::exception& e) {
    std::cerr << "Xpress error during construction: " << e.what() << std::endl;
    if (prob_) { XPRSdestroyprob(prob_); prob_ = nullptr; }
    throw;
  }
}

XpressInterface::~XpressInterface() {
  if (prob_) {
    XPRSdestroyprob(prob_);
    prob_ = nullptr;
  }
}

// ---- load variable names ----
void XpressInterface::loadVarNames() {
  varNames_.resize(numVars);

  // First call: get required buffer size
  int nbytes = 0;
  XPRSgetnamelist(prob_, XPRS_NAMES_COLUMN, nullptr, 0, &nbytes, 0, numVars - 1);

  if (nbytes <= 0) {
    // Fallback: generate names like "C0", "C1", ...
    for (int i = 0; i < numVars; ++i) {
      varNames_[i] = "C" + std::to_string(i);
    }
    return;
  }

  // Second call: get the actual names (null-separated)
  std::vector<char> buf(nbytes + 1, '\0');
  xprsCheck(XPRSgetnamelist(prob_, XPRS_NAMES_COLUMN, buf.data(), nbytes + 1,
                             nullptr, 0, numVars - 1),
            "XPRSgetnamelist");

  // Parse null-separated names
  int idx = 0;
  const char* p = buf.data();
  while (idx < numVars) {
    varNames_[idx] = std::string(p);
    p += varNames_[idx].size() + 1;  // skip past null terminator
    ++idx;
  }
}

// ---- build nonant maps (same logic as Gurobi/CPLEX) ----
void XpressInterface::_buildNonantsMaps() {
  std::unordered_set<std::string> nonantSet;
  std::unordered_set<std::string> modelVarSet;

  for (const auto& [ndn, nd] : treeNodeDataMap) {
    const auto& currentList = nd.nonAnts;
    nonantSet.insert(currentList.begin(), currentList.end());
    nodeNonantCounts[ndn] = currentList.size();
  }

  for (int i = 0; i < numVars; ++i) {
    const std::string& vname = varNames_[i];
    varNameToIndex[vname] = i;
    if (nonantSet.count(vname)) {
      modelVarSet.insert(vname);
      nonantNameToIndex[vname] = i;
      nonantIndexToName[i] = vname;
    }
  }

  // Check for missing nonants
  std::unordered_set<std::string> difference;
  for (const auto& vname : nonantSet) {
    if (modelVarSet.find(vname) == modelVarSet.end()) {
      difference.insert(vname);
    }
  }
  if (!difference.empty()) {
    std::cerr << "Your solver may have failed to instantiate!" << std::endl;
    std::cerr << "The following variables were in the nonant json, but not the model:" << std::endl;
    for (const auto& vname : difference) {
      std::cerr << vname << std::endl;
    }
    throw std::runtime_error("Nonant json vs model var name mismatch for "
                             + nonantFileName + " " + mpsFileName);
  }

  // Build index lists per tree node
  for (const auto& [ndn, nd] : treeNodeDataMap) {
    for (const auto& vname : nd.nonAnts) {
      auto it = nonantNameToIndex.find(vname);
      if (it == nonantNameToIndex.end()) {
        throw std::runtime_error("Nonant not in model: " + vname);
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

  // Duplicate check
  std::unordered_set<int> seen;
  for (int vidx : nonantDenseIndexes) {
    if (!seen.insert(vidx).second) {
      throw std::runtime_error("Duplicate variable index in nonantDenseIndexes: "
                               + std::to_string(vidx));
    }
  }
}

// ---- fullObjectiveRebuild ----
void XpressInterface::fullObjectiveRebuild() {
  // Validation
  if (WOn && !WPtr) {
    throw std::runtime_error("Xpress fullObjectiveRebuild: WOn=true but WPtr is null");
  }
  if (proxOn && !zPtr) {
    throw std::runtime_error("Xpress fullObjectiveRebuild: proxOn=true but zPtr is null");
  }
  if (rho.size() != static_cast<size_t>(totalNonantCount)) {
    throw std::runtime_error("Xpress fullObjectiveRebuild: rho size mismatch vs totalNonantCount");
  }
  if (WPtr && WPtr->size() != static_cast<size_t>(totalNonantCount)) {
    throw std::runtime_error("Xpress fullObjectiveRebuild: W size mismatch vs totalNonantCount");
  }
  if (zPtr && zPtr->size() != static_cast<size_t>(totalNonantCount)) {
    throw std::runtime_error("Xpress fullObjectiveRebuild: z size mismatch vs totalNonantCount");
  }
  for (int vidx : nonantDenseIndexes) {
    if (vidx < 0 || vidx >= numVars) {
      throw std::runtime_error("Xpress fullObjectiveRebuild: nonant index out of range: "
                               + std::to_string(vidx));
    }
  }

  try {
    // 1. Build linear objective coefficients
    std::vector<double> obj(numVars);
    for (int j = 0; j < numVars; ++j) {
      obj[j] = fullC[j];
    }

    // Overlay nonant deltas
    int pos = 0;
    for (int vidx : nonantDenseIndexes) {
      double delta = 0.0;
      if (WOn)   delta += (*WPtr)[pos];
      if (proxOn) delta -= rho[pos] * (*zPtr)[pos];
      obj[vidx] += delta;
      ++pos;
    }

    // Set all linear coefficients at once
    std::vector<int> allcols(numVars);
    std::iota(allcols.begin(), allcols.end(), 0);
    xprsCheck(XPRSchgobj(prob_, numVars, allcols.data(), obj.data()),
              "XPRSchgobj");

    // 2. Set quadratic diagonal terms
    // Xpress uses 0.5 * x^T Q x, so Q[j][j] = 2 * fullQDiag[j] for original terms.
    // For proximal: contribution is (rho/2) * x^2, so Q[j][j] += rho.
    for (int j = 0; j < numVars; ++j) {
      double qval = 2.0 * fullQDiag[j];  // original Q diagonal (Xpress convention)

      auto it = varToNonant.find(j);
      if (it != varToNonant.end() && proxOn) {
        qval += rho[it->second];  // proximal: 0.5 * rho * x^2
      }

      // Only call XPRSchgqobj for variables that need quadratic terms
      if (qval != 0.0 || fullQDiag[j] != 0.0 ||
          (it != varToNonant.end() && proxOn)) {
        xprsCheck(XPRSchgqobj(prob_, j, j, qval), "XPRSchgqobj");
      }
    }

    rhoChanged = false;

  } catch (const std::runtime_error& e) {
    std::cerr << "Exception in Xpress fullObjectiveRebuild: " << e.what() << std::endl;
    numericalBailout.store(true, std::memory_order_relaxed);
    return;
  }
}

void XpressInterface::applyBoundsToSolver() {
  for (int pos = 0; pos < totalNonantCount; ++pos) {
    int vidx = nonantDenseIndexes[pos];
    int cols[2] = {vidx, vidx};
    char btype[2] = {'L', 'U'};
    double bval[2] = {nonantLowerBounds[pos], nonantUpperBounds[pos]};
    xprsCheck(XPRSchgbounds(prob_, 2, cols, btype, bval), "XPRSchgbounds");
  }
}

double XpressInterface::getVarLB(int varIdx) const {
  double lb = 0.0;
  xprsCheck(XPRSgetlb(prob_, &lb, varIdx, varIdx), "XPRSgetlb");
  return lb;
}

double XpressInterface::getVarUB(int varIdx) const {
  double ub = 0.0;
  xprsCheck(XPRSgetub(prob_, &ub, varIdx, varIdx), "XPRSgetub");
  return ub;
}

// ---- solve ----
void XpressInterface::fetchSolution() const {
  int ncols = 0;
  XPRSgetintattrib(prob_, XPRS_COLS, &ncols);
  solution_.resize(ncols);

  // XPRSgetsolution works for both LP and MIP
  int solstatus = 0;
  XPRSgetsolution(prob_, &solstatus, solution_.data(), 0, ncols - 1);
}

objAndBnd XpressInterface::solve() {
  double obj = 0.0;
  double bound = 0.0;

  interrupted.store(false, std::memory_order_relaxed);
  if (abortRequested()) {
    interrupted.store(true, std::memory_order_relaxed);
    return objAndBnd{0.0, 0.0};
  }
  struct SolveScope {
    std::atomic<bool>& flag;
    ~SolveScope() { flag.store(false, std::memory_order_relaxed); }
  } solveScope{solveInProgress};
  solveInProgress.store(true, std::memory_order_relaxed);

  // Detect a user interrupt (interruptSolve) after an optimize call; the
  // caller must check wasInterrupted() and discard the result.
  auto userStopped = [&]() {
    int stopstatus = 0;
    xprsCheck(XPRSgetintattrib(prob_, XPRS_STOPSTATUS, &stopstatus),
              "XPRSgetintattrib(STOPSTATUS)");
    return stopstatus == XPRS_STOP_USER;
  };

  try {
    if (isMIP_) {
      xprsCheck(XPRSmipoptimize(prob_, ""), "XPRSmipoptimize");
      if (userStopped()) {
        interrupted.store(true, std::memory_order_relaxed);
        return objAndBnd{0.0, 0.0};
      }

      int mipstatus = 0;
      xprsCheck(XPRSgetintattrib(prob_, XPRS_MIPSTATUS, &mipstatus),
                "XPRSgetintattrib(MIPSTATUS)");

      if (mipstatus == XPRS_MIP_OPTIMAL || mipstatus == XPRS_MIP_SOLUTION) {
        xprsCheck(XPRSgetdblattrib(prob_, XPRS_MIPOBJVAL, &obj),
                  "XPRSgetdblattrib(MIPOBJVAL)");
        xprsCheck(XPRSgetdblattrib(prob_, XPRS_BESTBOUND, &bound),
                  "XPRSgetdblattrib(BESTBOUND)");
      } else if (mipstatus == XPRS_MIP_INFEAS) {
        throw std::runtime_error("Model infeasible (MIP) for " + scenarioName);
      } else if (mipstatus == XPRS_MIP_UNBOUNDED) {
        throw std::runtime_error("Model unbounded (MIP) for " + scenarioName);
      } else if (mipstatus == XPRS_MIP_NO_SOL_FOUND) {
        throw std::runtime_error("No MIP solution found for " + scenarioName +
                                 " (MIPSTATUS=" + std::to_string(mipstatus) + ")");
      } else {
        throw std::runtime_error("Unexpected MIP status " + std::to_string(mipstatus)
                                 + " for " + scenarioName);
      }
    } else {
      xprsCheck(XPRSlpoptimize(prob_, ""), "XPRSlpoptimize");
      if (userStopped()) {
        interrupted.store(true, std::memory_order_relaxed);
        return objAndBnd{0.0, 0.0};
      }

      int lpstatus = 0;
      xprsCheck(XPRSgetintattrib(prob_, XPRS_LPSTATUS, &lpstatus),
                "XPRSgetintattrib(LPSTATUS)");

      if (lpstatus == XPRS_LP_OPTIMAL) {
        xprsCheck(XPRSgetdblattrib(prob_, XPRS_LPOBJVAL, &obj),
                  "XPRSgetdblattrib(LPOBJVAL)");
        bound = obj;  // LP: bound == obj at optimality
      } else if (lpstatus == XPRS_LP_INFEAS) {
        throw std::runtime_error("Model infeasible (LP) for " + scenarioName);
      } else if (lpstatus == XPRS_LP_UNBOUNDED) {
        throw std::runtime_error("Model unbounded (LP) for " + scenarioName);
      } else if (lpstatus == XPRS_LP_UNFINISHED) {
        // Time limit or iteration limit — try to get best available
        std::cerr << "Warning: Xpress LP unfinished for '" << scenarioName
                  << "'; using best available solution\n";
        xprsCheck(XPRSgetdblattrib(prob_, XPRS_LPOBJVAL, &obj),
                  "XPRSgetdblattrib(LPOBJVAL)");
        bound = obj;
      } else {
        throw std::runtime_error("Unexpected LP status " + std::to_string(lpstatus)
                                 + " for " + scenarioName);
      }
    }

    // Fetch solution into solution_ vector
    fetchSolution();

  } catch (const std::runtime_error& e) {
    std::cerr << "Xpress solve error: " << e.what() << std::endl;
    throw;
  }

  return objAndBnd{obj, bound};
}

// Called from the control thread to stop a solve in flight (best effort).
// XPRSinterrupt is documented as safe to call from another thread; the
// solveInProgress guard keeps a request that lands between solves from
// affecting a later solve.
void XpressInterface::interruptSolve() {
  if (solveInProgress.load(std::memory_order_relaxed) && prob_) {
    XPRSinterrupt(prob_, XPRS_STOP_USER);
  }
}

// ---- solution extraction ----
std::vector<double> XpressInterface::copyNonantSolution() const {
  std::vector<double> returnVector;
  for (const auto& [nd, indexList] : nonantIndexes) {
    for (const auto& i : indexList) {
      returnVector.push_back(solution_[i]);
    }
  }
  return returnVector;
}

std::vector<double> XpressInterface::copyEntireSolution() const {
  return std::vector<double>(solution_.begin(), solution_.begin() + numVars);
}

void XpressInterface::extractNonantSolution(std::vector<double>& nonantSolution) const {
  if (nonantSolution.size() != static_cast<size_t>(totalNonantCount)) {
    std::cerr << "nonantSolution.size()=" << nonantSolution.size()
              << " totalNonantCount=" << totalNonantCount << std::endl;
    throw std::runtime_error("bad call to extractNonantSolution");
  }
  int j = 0;
  for (const auto& [nd, indexList] : nonantIndexes) {
    for (const auto& i : indexList) {
      nonantSolution[j++] = solution_[i];
    }
  }
}

void XpressInterface::extractNodeSolution(const std::string nodeName,
                                           std::vector<double>& nodeSolution) const {
  auto nI = nonantIndexes.at(nodeName);
  if (nodeSolution.size() != nI.size()) {
    std::cerr << "nodeSolution.size()=" << nodeSolution.size()
              << " nonantIndex[nodeName].size()=" << nI.size() << std::endl;
    throw std::runtime_error("bad call to extractNodeSolution");
  }
  int j = 0;
  for (const auto& i : nI) {
    nodeSolution[j++] = solution_[i];
  }
}

// ---- parameter setters ----
void XpressInterface::setXpressIntControl(int control, int value) {
  int ret = XPRSsetintcontrol(prob_, control, value);
  if (ret != 0) {
    std::cerr << "Warning: Xpress setintcontrol(" << control << ", " << value
              << ") failed with code " << ret << std::endl;
  }
}

void XpressInterface::setXpressDblControl(int control, double value) {
  int ret = XPRSsetdblcontrol(prob_, control, value);
  if (ret != 0) {
    std::cerr << "Warning: Xpress setdblcontrol(" << control << ", " << value
              << ") failed with code " << ret << std::endl;
  }
}

// ---- print / export ----
void XpressInterface::printSolution() {
  std::cout << "Subproblem name=" << scenarioName << std::endl;
  double objVal = 0.0;
  if (isMIP_) {
    XPRSgetdblattrib(prob_, XPRS_MIPOBJVAL, &objVal);
  } else {
    XPRSgetdblattrib(prob_, XPRS_LPOBJVAL, &objVal);
  }
  std::cout << "Optimal objective: " << objVal << std::endl;
  for (int i = 0; i < numVars; ++i) {
    std::cout << varNames_[i] << " = " << solution_[i] << std::endl;
  }
}

void XpressInterface::nonantSolutionVectorToCsv(std::string fileName) const {
  std::ofstream outFile(fileName);
  if (outFile.is_open()) {
    for (const auto& [name, i] : nonantNameToIndex) {
      outFile << name << "," << solution_[i] << "\n";
    }
    outFile.close();
  } else {
    throw std::runtime_error("Error trying to write file=" + fileName);
  }
}

void XpressInterface::exportModel(const std::string fname) {
  // Determine format from extension
  std::string flags = "";
  if (fname.size() >= 3 && fname.substr(fname.size() - 3) == ".lp") {
    flags = "l";
  }
  xprsCheck(XPRSwriteprob(prob_, fname.c_str(), flags.c_str()),
            "XPRSwriteprob");
}

bool XpressInterface::setStartAtCurrent() {
  // Load the current LP solution as a warm start
  if (solution_.empty()) return false;

  // For LP problems, XPRSloadlpsol can provide a warm start
  int ncols = 0;
  XPRSgetintattrib(prob_, XPRS_COLS, &ncols);
  if (static_cast<int>(solution_.size()) < ncols) return false;

  int status = 0;
  int ret = XPRSloadlpsol(prob_, solution_.data(), nullptr, nullptr, nullptr, &status);
  return (ret == 0);
}

void XpressInterface::setLogFile(const std::string fileName) {
  SolverInterface::setLogFile(fileName);
  if (prob_) {
    if (!fileName.empty()) {
      XPRSsetlogfile(prob_, fileName.c_str());
    } else {
      XPRSsetlogfile(prob_, nullptr);
      XPRSsetintcontrol(prob_, XPRS_OUTPUTLOG, XPRS_OUTPUTLOG_NO_OUTPUT);
    }
  }
}

void XpressInterface::clearModel() {
  if (prob_) {
    XPRSdestroyprob(prob_);
    prob_ = nullptr;
  }

  int ret = XPRScreateprob(&prob_);
  if (ret != 0 || !prob_) {
    throw std::runtime_error("XPRScreateprob failed in clearModel");
  }

  // Silence new problem
  XPRSsetintcontrol(prob_, XPRS_OUTPUTLOG, XPRS_OUTPUTLOG_NO_OUTPUT);

  numVars = 0;
  varNames_.clear();
  solution_.clear();
  isMIP_ = false;
}
