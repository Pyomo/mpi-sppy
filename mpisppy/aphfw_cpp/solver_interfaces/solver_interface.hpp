#ifndef SOLVER_INTERFACE_HPP
#define SOLVER_INTERFACE_HPP

// TBD List:
//    - Quadratic objectives are supported on the diagonal only (see fullQDiag
//      and hasOriginalQuadratic); cross terms x_i*x_j are detected and throw.
//    - Maybe put the maps and vars in a struct to get them all at once
//    - related: Improve loop efficiency, particularly the map building loops
//    - related: maybe standardize loop idioms a little more
//    - Think harder about references versus copies of var names in particular
// Notes for users:
//     - We expect the caller to use references to the W vector etc. to update it
//       so the caller needs to keep track of whether the model needs to be
//       solved.
//     - Important: the caller owns W and z. We handle rho differently.
//     - If the caller wants a squared term, the caller needs to call the add
//       function and also set rho.
// Notes for developers and users:
//     - Whenever we need solution values, we get them from the solver
//       to be sure the values are fresh. (In normal operation, that will be
//       done only once per iteration of the caller.)
//     - The logfile is created, timestamped, and closed here. Children
//       should append it with solver logs.

/* Overloaded words: "variable" and "index"

I use the "var" to refer to solver (e.g. cplex) objects for model
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

The values for dense vectors W, rho, and z are supplied by the
caller. A pointer to W and z vectors should be obtained from the
caller and then the caller can change the values. To send the new
values to the solver, the caller *might* want to call
something to update the objective. The values for rho need to be swapped out using
a setter call.

(Note as of Feb 20, 2026: I might be calling fullObjectiveRebuild too often.)
*/

#include <atomic>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <optional>
#include <unordered_set>
#include "json.hpp"
#include "../utils/utils.hpp"
#include "TreeNodeData.hpp"

// ‘nlohmann::json_abi_v3_11_3::json’?
using json = nlohmann::json;
using NodeDataMap = std::map<std::string, TreeNodeData>;
inline void from_json(const json& j, TreeNodeData& nodeData) {
    j.at("serialNumber").get_to(nodeData.serialNumber);
    j.at("condProb").get_to(nodeData.condProb);
    j.at("nonAnts").get_to(nodeData.nonAnts);
}
// map from node name to var indexes for nonants at the node
using NonantTreeIndexMap = std::map<std::string, std::vector<int>>;

struct objAndBnd {
  double obj;
  double bnd;
};

class NLCoeffs;  // forward

//===============================================

class SolverInterface {
  friend class NLCoeffs;  // special purpose helper class for NL children
public:
  SolverInterface(const std::string& scenarioName);
  virtual ~SolverInterface() = default;
  virtual void pprint(std::ostream& out) const;
  
  // this class and it's children are move-only
  SolverInterface(const SolverInterface&) = delete;
  SolverInterface& operator=(const SolverInterface&) = delete;
  SolverInterface(SolverInterface&&) = default;
  SolverInterface& operator=(SolverInterface&&) = default;

  virtual objAndBnd solve() = 0;
  bool hasNumericalBailout() const { return numericalBailout.load(std::memory_order_relaxed); }

  // Cooperative abort support.  The controller owns an abort flag; once it is
  // attached, solve() returns immediately (with wasInterrupted() true) when
  // the flag is already set, and interruptSolve() may be called from another
  // thread to stop a solve that is in flight.  Interrupted solves must not
  // throw; the caller is expected to discard their results.
  void attachAbortFlag(const std::atomic<bool>* flag) { abortFlagPtr = flag; }
  bool abortRequested() const {
    return abortFlagPtr && abortFlagPtr->load(std::memory_order_relaxed);
  }
  bool wasInterrupted() const { return interrupted.load(std::memory_order_relaxed); }
  virtual void interruptSolve() {}  // async-safe, best-effort; default no-op
  std::string getScenarioName() const {
    return scenarioName;
  }
  virtual std::string getVarName(const size_t i) const = 0; // all vars
  virtual std::vector<double> copyNonantSolution() const = 0;
  virtual std::vector<double> copyEntireSolution() const = 0;
  virtual void extractNonantSolution(std::vector<double>& nonantSolution) const = 0;
  virtual void extractNodeSolution(const std::string nodeName,
                           std::vector<double>& nodeSolution) const = 0;
  int getNodeNonantCount(const std::string& nodeName) const {
    return nodeNonantCounts.at(nodeName);
  }
  // canonical, solver-specific full rebuild
  virtual void fullObjectiveRebuild() = 0;

  // Variable bounds on nonant variables (for reduced-costs bridge)
  void setNonantBounds(const std::vector<double>& lb, const std::vector<double>& ub);
  void captureInitialBounds();  // read bounds from solver once after model load
  const std::vector<double>& getNonantLowerBounds() const { return nonantLowerBounds; }
  const std::vector<double>& getNonantUpperBounds() const { return nonantUpperBounds; }

protected:
  // Apply stored bounds to the underlying solver model.  Pure virtual.
  virtual void applyBoundsToSolver() = 0;
  virtual double getVarLB(int varIdx) const = 0;
  virtual double getVarUB(int varIdx) const = 0;

public:
  // Scale all original objective coefficients by the given factor.
  // Must be called before any solves; caller should call fullObjectiveRebuild() after.
  void scaleObjective(double factor) {
    for (auto& c : fullC) c *= factor;
  }

  // Objective cutoff constraint for FW inner loop (only used by Grad subclasses).
  // Adds: current_objective <= cutoffValue.  Returns true if added.
  // While cutoff is active, infeasibility is expected (no improving direction).
  virtual bool addObjectiveCutoff(double /*cutoffValue*/) { return false; }
  virtual void removeObjectiveCutoff() {}
  bool isInfeasibleDueToCutoff() const { return infeasibleDueToCutoff_; }
  void addXSquared();  // we are just going to "rebuild" the objective
  double computeObjFctVal(const std::vector<double>& fullX) const;
  double computeOrigObjFctVal(const std::vector<double>& fullX) const;  
  // The algorithmic (W + prox) part of computeObjFctVal, over nonants only, so
  // that a caller ranking many columns under a fixed model can cache the
  // original part -- which does not depend on W, z or rho -- and pay only the
  // nonant-length loop per column.  computeObjFctVal is the sum of the two.
  double computeNonantPenalty(const std::vector<double>& fullX) const;
  // this needs to be in order
  std::vector<std::string> nonantNamesVector() const {
    std::vector<std::string> retval;
    retval.reserve(nonantNameToIndex.size());
    // working with solver indexes
    for (int i : nonantDenseIndexes) {
      auto it = nonantIndexToName.find(i);
      if (it != nonantIndexToName.end()) {
        retval.push_back(it->second);
      } else {
        throw std::runtime_error("Missing name for nonant var index: " + std::to_string(i));
      }
    }
    return retval;
  }
  virtual void logMsg(std::string msg) const {
    if ( !logFileName.empty() ) {      
      std::ofstream ofs(logFileName, std::ios::app);
      if (!ofs) {
        throw std::runtime_error("Failed to open file: " + logFileName);
      }
      ofs << msg << std::endl;
    }
  }
  // Algorithmic data
  // Note: the caller allocates the memory, callers should change the values
  std::vector<double>& get_W() { return *WPtr; }  // ? is get needed for z and W?
  std::vector<double>& get_z() { return *zPtr; }
  std::vector<double>* get_WPtr() { return WPtr; }  // this might not be needed?
  std::vector<double>* get_zPtr() { return zPtr; }
  void set_WOn (const bool newWOn);
  void set_proxOn (const bool newproxOn);
  // rho is an exception due to cplex and quadratics (it is the x^2 coefficient)
  //   Note that this doesn't return a reference. Use rho setter to change it.
  std::vector<double> get_rho() { return rho; }
  virtual void setLogFile(const std::string fileName);
  std::string getLogFileName() const { return logFileName; }
  const std::vector<int>& getNodeIndexes(const std::string nodeName) const {
    return nonantIndexes.at(nodeName);
  }

// read-only access to solver data (no copies)
  const std::vector<double>& get_fullC_const() const noexcept { return fullC; }
  const std::vector<double>& get_fullQDiag_const() const noexcept { return fullQDiag; }
  const std::vector<int>&    get_nonantDenseIndexes_const() const noexcept { return nonantDenseIndexes; }

  // const getters
  const std::vector<double>& get_W_const() const {
    if (!WPtr) throw std::runtime_error("get_W_const: WPtr is null");
    return *WPtr;
  }
  const std::vector<int>& getNonantDenseIndexesConst() const {
    return nonantDenseIndexes;
  }  

  const std::vector<double>& get_z_const() const {
    if (!zPtr) throw std::runtime_error("get_z_const: zPtr is null");
    return *zPtr;
  }  
  // from here down are functions the caller should rarely, if ever, need.
  virtual void printSolution() = 0;
  void printNodeDataMaps();
  void printNonantSolutionVector();
  virtual void nonantSolutionVectorToCsv(std::string fileName) const = 0;
  std::string nonantPositionToName(int pos) const;
  virtual void exportModel(const std::string fname) = 0;  // have an mps file name suffix
  virtual bool setStartAtCurrent() = 0;  // sets the starting soln using current soln, if possible

  // A utility to let the caller create vectors to match nonants (W, e.g.)
  std::vector<double> createZerosToMatchNonants() const {
    return std::vector<double>(totalNonantCount, 0.0);
  }
  // These attach functions are intended to be called only once.
  void attach_W (std::vector<double>& newW);
  void attach_z (std::vector<double>& z);
  // rho is an exception due to cplex and quadratics (it is the x^2 coefficient)
  //    because we need to track its changes so as to remove terms from the 
  //    objective. I.e. you should use set_rho to change its value.
  void set_rho(const std::vector<double>& newrho);
  // Note for WOn and ProxOn, we are not providing references so use the setters
  bool isWOn() const { return WOn; }
  bool isProxOn() const { return proxOn; }
  bool hasRhoChanged() const { return rhoChanged; }
  void setRhoChanged(const bool nv) { rhoChanged = nv; }  // use with care
  
  int getTotalNonantCount() const { return totalNonantCount; }

  // True if the objective read from the model file has quadratic terms, i.e.
  // the original problem is nonlinear.  Set by the child constructors when
  // they scan the objective.
  bool hasOriginalQuadratic() const noexcept { return hasOrigQuad; }

  const NodeDataMap& getTreeNodeDataMap() const {return treeNodeDataMap;}
  double getProbability() const {return probability;}
  int getGlobalNodeCount() const {return globalNodeCount;}
  size_t iToNai(const int i) const {
    // for a var index i, give the nonant index or -1
    if (auto it = varToNonant.find(i); it != varToNonant.end()) {
      return it->second;
    }
   return -1;
  }

  // a function for debugging and testing:
  double computeWdotX_fromFullX(const std::vector<double>& fullX) const {
    if (!WPtr) throw std::runtime_error("computeWdotX_fromFullX: WPtr is null");
    if (fullX.size() != fullC.size())
      throw std::runtime_error("computeWdotX_fromFullX: fullX size mismatch with fullC");

    const auto& W = *WPtr;
    if (W.size() != nonantDenseIndexes.size())
      throw std::runtime_error("computeWdotX_fromFullX: W size mismatch with nonantDenseIndexes");

    double Wx = 0.0;
    for (size_t pos = 0; pos < nonantDenseIndexes.size(); ++pos) {
      int vidx = nonantDenseIndexes[pos];
      Wx += W[pos] * fullX[(size_t)vidx];
    }
    return Wx;
  }  
  
protected:
  // Set by addObjectiveCutoff/removeObjectiveCutoff and checked by solve()
  bool tolerateInfeasible_ = false;
  bool infeasibleDueToCutoff_ = false;

  // constructor arg
  std::string scenarioName;

  // for solver and other logs
  std::string logFileName;
  
  // file names that define the scenario data files
  std::string mpsFileName;
  std::string nonantFileName;

  // scenario data (we will not have a name field)
  double probability;

  // nonants tree map (keys are tree nodes; data are var name lists)
  // The order in these lists establishes the nonant ordering.
  NodeDataMap treeNodeDataMap;
  NonantTreeIndexMap nonantIndexes;  // the indexes are into vars.

  // from the json scenario file; useful for creating comms
  int globalNodeCount;   // number of nodes in the entire tree (from json)
  

  // maps between nonant names and solver indexes
  std::map<std::string, int> nonantNameToIndex;
  std::map<int, std::string> nonantIndexToName;
  std::map<std::string, int> varNameToIndex;  // all vars

  // Other nonant data
  int totalNonantCount = 0;
  std::map<std::string, int> nodeNonantCounts;
  std::vector<double> fullC;  // created by children; should not change
  std::vector<int> nonantDenseIndexes; // ordered indexes into model vars; made by children
  // only has a key (which is a var index) for nonants, gives the nonant index
  std::map<int, int> varToNonant; 
  // Algorithmic data
  std::vector<double> rho;
  std::vector<double> oldRho;
  std::vector<double> nonantLowerBounds;
  std::vector<double> nonantUpperBounds;
  std::vector<double>* WPtr = 0;
  std::vector<double>* zPtr = 0;
  bool WOn = false;
  bool proxOn = false;
  bool rhoChanged = false;
  std::atomic<bool> numericalBailout{false};

  // Cooperative abort state (see the public abort block above)
  const std::atomic<bool>* abortFlagPtr = nullptr;
  std::atomic<bool> interrupted{false};
  std::atomic<bool> solveInProgress{false};

  // Original quadratic objective terms read from the file.
  // Diagonal-only initial support: q_i * x_i^2
  std::vector<double> fullQDiag;   // size = #vars, aligned with vars[] like fullC
  bool hasOrigQuad = false;        // any quadratic terms found
  bool hasOrigQuadCross = false;   // x_i*x_j with i != j detected
  

  // let a child class (e.g. NL) start over with the underlying solver
  virtual void clearModel() = 0;
  
  // local functions 
  void loadNonantsFromJson();
  inline void _vsetter(std::vector<double>*& v, std::vector<double>& newv);
};

#endif // SOLVER_INTERFACE_HPP
