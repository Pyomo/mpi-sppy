#include "solver_interface.hpp"
#include <unordered_set>
#include <string>
#include <iostream>
#include <exception>
#include <chrono>
//#include <format>

SolverInterface::SolverInterface(const std::string& scenarioName):
  scenarioName(scenarioName),
  mpsFileName(scenarioName + ".mps"),
  nonantFileName(scenarioName + "_nonants.json")
{
  // nonants
  loadNonantsFromJson();
  // children must build the maps
  
  // Allocate memory for algorithm data; TBD think harder about rho
  rho.assign(totalNonantCount, 0.0);
  oldRho.assign(totalNonantCount, 0.0);
};

double SolverInterface::computeObjFctVal(const std::vector<double>& fullX) const {
  // Defensive diagnostics to avoid SIGSEGV and to reveal root cause.
  // If something is inconsistent, print sizes and throw runtime_error.

  // Basic size check against fullC
  if (fullX.size() != fullC.size()) {
    std::ostringstream os;
    os << "computeObjFctVal: fullX size (" << fullX.size()
       << ") != fullC size (" << fullC.size() << ")";
    std::cerr << "[FATAL] " << os.str() << std::endl;

    // Extra helpful info
    std::cerr << "[FATAL] nonantDenseIndexes.size()=" << nonantDenseIndexes.size()
              << " totalNonantCount=" << totalNonantCount
              << " varToNonant.size()=" << varToNonant.size() << std::endl;

    throw std::runtime_error(os.str());
  }

  // 1) + 2) linear part and original diagonal quadratic
  // 3) algorithmic prox / W terms over nonants
  return computeOrigObjFctVal(fullX) + computeNonantPenalty(fullX);
}

// The W and prox terms of the subproblem objective, over nonants only.  Split
// out of computeObjFctVal so that a caller ranking a growing column set (see
// FrankWolfe::setVertexExport) can cache the original objective per column and
// re-evaluate only this part when W, z or rho move.
double SolverInterface::computeNonantPenalty(const std::vector<double>& fullX) const {
  // Check WPtr and zPtr presence and sizes
  if (WPtr == nullptr || zPtr == nullptr) {
    std::ostringstream os;
    os << "computeNonantPenalty: WPtr or zPtr is null (WPtr=" << WPtr << ", zPtr=" << zPtr << ")";
    std::cerr << "[FATAL] " << os.str() << std::endl;
    throw std::runtime_error(os.str());
  }
  if (WPtr->size() != (size_t) totalNonantCount || zPtr->size() != (size_t) totalNonantCount) {
    std::ostringstream os;
    os << "computeNonantPenalty: WPtr/zPtr size mismatch: WPtr.size()=" << WPtr->size()
       << " zPtr.size()=" << zPtr->size()
       << " expected totalNonantCount=" << totalNonantCount;
    std::cerr << "[FATAL] " << os.str() << std::endl;
    throw std::runtime_error(os.str());
  }

  double penalty = 0.0;

  // Use varToNonant mapping: map solver var index -> nonant position (nai)
  for (size_t pos = 0; pos < nonantDenseIndexes.size(); ++pos) {
    int vidx = nonantDenseIndexes[pos];   // solver var index
    if (vidx < 0 || (size_t)vidx >= fullX.size()) {
      std::ostringstream os;
      os << "computeNonantPenalty: nonantDenseIndexes[" << pos << "] = " << vidx
         << " is out of range for fullX.size()=" << fullX.size();
      std::cerr << "[FATAL] " << os.str() << std::endl;
      throw std::runtime_error(os.str());
    }

    // get nai from varToNonant (should exist)
    auto it = varToNonant.find(vidx);
    if (it == varToNonant.end()) {
      std::ostringstream os;
      os << "computeNonantPenalty: varToNonant missing mapping for var index " << vidx;
      std::cerr << "[FATAL] " << os.str() << std::endl;
      throw std::runtime_error(os.str());
    }
    int nai = it->second;
    if (nai < 0 || nai >= totalNonantCount) {
      std::ostringstream os;
      os << "computeNonantPenalty: derived nai " << nai << " out of range [0," << totalNonantCount-1 << "]";
      std::cerr << "[FATAL] " << os.str() << std::endl;
      throw std::runtime_error(os.str());
    }

    double dx = fullX[(size_t)vidx] - (*zPtr)[(size_t)nai];
    penalty += (*WPtr)[(size_t)nai] * dx
             + 0.5 * rho[(size_t)nai] * dx * dx;
  }

  return penalty;
}

double SolverInterface::computeOrigObjFctVal(const std::vector<double>& fullX) const {
  // Compute *original* objective (linear + original diagonal quadratic only).
  // Defensive checks to help catch mismatches that previously caused linker/runtime issues.

  if (fullX.size() != fullC.size()) {
    std::ostringstream os;
    os << "computeOrigObjFctVal: fullX size (" << fullX.size()
       << ") != fullC size (" << fullC.size() << ")";
    std::cerr << "[FATAL] " << os.str() << std::endl;
    throw std::runtime_error(os.str());
  }

  double val = 0.0;

  // linear part
  for (size_t i = 0; i < fullC.size(); ++i) {
    val += fullC[i] * fullX[i];
  }

  // original diagonal quadratic part (if any)
  if (!fullQDiag.empty()) {
    // guard against slight size mismatches by using the minimum size
    size_t minsz = std::min(fullQDiag.size(), fullX.size());
    for (size_t i = 0; i < minsz; ++i) {
      double q = fullQDiag[i];
      if (q != 0.0) {
        val += q * fullX[i] * fullX[i];
      }
    }
    if (fullQDiag.size() != fullX.size()) {
      std::cerr << "[WARN] computeOrigObjFctVal: fullQDiag.size()="
                << fullQDiag.size() << " differs from fullX.size()="
                << fullX.size() << " (used min=" << minsz << ")\n";
    }
  }

  return val;
}

void SolverInterface::setLogFile(const std::string fileName) {
  // if you bother to send in an empty sring, we assume you want no output
  logFileName = fileName;
  if ( !fileName.empty() ) {
    try {
      // Open the file for writing (it will close when out of scope)
      std::ofstream ofs(fileName);
      if (!ofs) {
        throw std::runtime_error("Failed to open file: " + fileName);
      }

      const auto now = std::chrono::system_clock::now();
      const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
      ofs << std::ctime(&t_c);
    
    } catch (const std::exception& ex) {
      std::cerr << "Error writing current date and time to log file: "
                << ex.what() << std::endl;
    }
  }
}

void SolverInterface::loadNonantsFromJson() {
  // Called by the constructor; also set totalNonantCount
  // Tree Information from json to the private map
  // Also get the probability
  std::ifstream jsonFile(nonantFileName);
  if (!jsonFile.is_open()) {
    throw std::runtime_error("Failed to open file: "+ nonantFileName);
  }
  // Parse the JSON file
  json jsonData;
  try {
    jsonFile >> jsonData;
  } catch (const std::exception& e) {
    std::cerr << "Error parsing JSON: " << e.what() << std::endl;
    throw std::runtime_error("The error was in: "+ nonantFileName);
  }
  probability = jsonData["scenarioData"]["scenProb"];
  globalNodeCount = jsonData["treeData"]["globalNodeCount"];
  try {
    treeNodeDataMap = jsonData["treeData"]["nodes"].get<NodeDataMap>();
  } catch (const std::exception& e) {
    std::cerr << "Error converting JSON to map: " << e.what() << std::endl;
    throw std::runtime_error("The error was in: "+ nonantFileName);
  }
  totalNonantCount = 0;
  for (const auto& [ndn, nd] : treeNodeDataMap) {
    totalNonantCount += nd.nonAnts.size();
  }
}

// pass by reference so we actually do the desired assignment
inline void SolverInterface::_vsetter(std::vector<double>*& v,
                                     std::vector<double>& newv) {
  if (newv.size() != totalNonantCount) {
    throw std::invalid_argument("Algorithm nonant parameter size mismatch: "
                                + std::to_string(newv.size()) + " != "
                                + std::to_string(totalNonantCount));
    }
  v = &newv;
}

void SolverInterface::set_rho(const std::vector<double>& newrho) {
  // rho is special for cplex because it is the quadratic coefficient
  // so we do an elementwise copy and keep track of the old rho
  if (newrho.size() != totalNonantCount) {
    throw std::invalid_argument("Size mismatch in set_rho: new="
                                + std::to_string(newrho.size())
                                + " nonants=" + std::to_string(totalNonantCount));
  }
  oldRho = rho;
  rho = newrho;
  rhoChanged = true;
}
void SolverInterface::setNonantBounds(const std::vector<double>& lb,
                                      const std::vector<double>& ub) {
  if (static_cast<int>(lb.size()) != totalNonantCount ||
      static_cast<int>(ub.size()) != totalNonantCount) {
    throw std::invalid_argument("Size mismatch in setNonantBounds: lb="
      + std::to_string(lb.size()) + " ub=" + std::to_string(ub.size())
      + " nonants=" + std::to_string(totalNonantCount));
  }
  nonantLowerBounds = lb;
  nonantUpperBounds = ub;
  applyBoundsToSolver();
}

void SolverInterface::captureInitialBounds() {
  nonantLowerBounds.resize(totalNonantCount);
  nonantUpperBounds.resize(totalNonantCount);
  int pos = 0;
  for (int vidx : nonantDenseIndexes) {
    nonantLowerBounds[pos] = getVarLB(vidx);
    nonantUpperBounds[pos] = getVarUB(vidx);
    ++pos;
  }
}

void SolverInterface::attach_W (std::vector<double>& newW) {
  _vsetter(WPtr, newW);
}
void SolverInterface::attach_z (std::vector<double>& newz) {
  _vsetter(zPtr, newz);
}
void SolverInterface::set_WOn (const bool newWOn) {
  if (WOn != newWOn) {
    if (newWOn && !WOn) {
      WOn = newWOn;  // i.e. true
    } else {
      // It could be supported, but I don't see a use case
      throw std::invalid_argument("WOn from true to false not supported");
    }
  }
}


void SolverInterface::set_proxOn (const bool newProxOn) {
  if (proxOn != newProxOn) {
    if (newProxOn && !proxOn) {
      proxOn = newProxOn;
    } else {
      // It could be supported, but I don't see a use case
      throw std::invalid_argument("ProxOn from true to false not supported");
    }
  }
}


void SolverInterface::addXSquared() {
  // “adding x^2” means: ensure prox quadratic is in the objective.
  // We do that by rebuilding the objective.
  if (!proxOn) {
    throw std::runtime_error("addXSquared called but proxOn=false (call set_proxOn(true) first)");
  }
  if (rho.size() != static_cast<size_t>(totalNonantCount)) {
    throw std::runtime_error("addXSquared: rho size mismatch vs totalNonantCount");
  }
  // fullObjectiveRebuild will also require W/z depending on flags;
  // don’t enforce W/z here—let fullObjectiveRebuild decide.
  fullObjectiveRebuild();
}


void SolverInterface::printNodeDataMaps() {
  std::cout << "Tree nodes: [nonant names]" << std::endl;
  for (const auto& [ndn, nd] : treeNodeDataMap) {
    std::cout << ndn << ": ";
    for (const std::string& vname : nd.nonAnts) {
      std::cout << vname << " ";
    }
    std::cout << std:: endl;
  }
  std::cout << "Tree nodes: [nonant solver indexes]" << std::endl;
  printListMap(nonantIndexes);
}

std::string SolverInterface::nonantPositionToName(int pos) const {
  // pos is the position in dense nonant vectors
  if ((pos < 0) || (pos > totalNonantCount)) {
    throw std::invalid_argument("nonantPositionToName called with " +
                                std::to_string(pos) + "(not in [0," +
                                std::to_string(totalNonantCount) + "])");
                                
  }
  std::string cnd;
  int sofar = 0;
  for (const auto& [nd, count]: nodeNonantCounts) {
    if (pos < sofar+count) {
      cnd = nd;
      break;
    }
    sofar += count;
  }
  return nonantIndexToName.at(nonantIndexes.at(cnd).at(pos-sofar));
}


void SolverInterface::printNonantSolutionVector() {
  auto sa = copyNonantSolution();
  std::cout << "Nonant solution vector without labels" << std::endl;
  for (const auto& value : sa) {
      std::cout << value << " ";
  }
  std::cout << std::endl;
}


void SolverInterface::pprint(std::ostream& out) const {
  out << "===== SolverInterface: " << scenarioName << " =====\n";

  // Log file info
  out << "  Log file: " << (logFileName.empty() ? "(none)" : logFileName) << "\n";

  // Total count
  out << "  Total nonant count: " << totalNonantCount << "\n";

  // Nonant name/index mappings
  out << "  Nonant variable mappings:\n";
  for (const auto& [name, index] : nonantNameToIndex) {
    out << "    " << name << " -> " << index << "\n";
  }

  // Tree structure
  out << "  Nonant tree structure:\n";
  for (const auto& [nodeName, node] : treeNodeDataMap) {
    out << "    Node: " << nodeName << "\n";
    out << "      Conditional Probability: " << node.condProb << "\n";
    out << "      Non-anticipative variables: ";
    for (const auto& var : node.nonAnts) {
      out << var << " ";
    }
    out << "\n";
  }

  // Nonant indexes (mapping from node name to index list)
  out << "  Nonant indexes (into solver vars, by node):\n";
  for (const auto& [nodeName, idxList] : nonantIndexes) {
    out << "    Node " << nodeName << ": ";
    for (int idx : idxList) {
      out << idx << " ";
    }
    out << "\n";
  }

  out << "  Nonant Dense indexes (into the solver vars):\n";
  std::for_each(nonantDenseIndexes.begin(),
                nonantDenseIndexes.end(), [&](int i) {
                  out << i << " ";});  
  out << "\n";

  
  // Node-specific nonant counts
  out << "  Node nonant counts:\n";
  for (const auto& [node, count] : nodeNonantCounts) {
    out << "    " << node << ": " << count << "\n";
  }

  // Vector sizes
  out << "  Vector sizes:\n";
  out << "    rho.size(): " << rho.size() << "\n";
  if (WPtr) {
    out << "    (*WPtr).size(): " << (*WPtr).size() << " (WOn=" << std::boolalpha << WOn << ")\n";
  } else {
    out << "     W pointer is null\n";
  }
  if (zPtr) {
    out << "    (*zPtr).size(): " << (*zPtr).size() << "\n";
  } else {
    out << "     z pointer is null\n";
  }
  out << "  ProxOn: " << std::boolalpha << proxOn << "\n";
  out << "  rhoChanged: " << std::boolalpha << rhoChanged << "\n";

  // Output full contents of vectors
  auto printVector = [&out](const std::string& label, const std::vector<double>& v) {
    out << "  " << label << ": [";
    for (size_t i = 0; i < v.size(); ++i) {
      out << v[i];
      if (i + 1 < v.size()) out << ", ";
    }
    out << "]\n";
  };

  out << "  --- Algorithmic Vectors ---\n";
  if (WPtr) printVector("W", *WPtr);
  if (zPtr) printVector("z", *zPtr);
  printVector("rho", rho);
  out << "===========================================\n";
}
