#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>

#include "groupController.hpp"   // defines subdomain + debugLevels
#include "scenSolveType.hpp"
#include "ProblemWithSolvers.hpp"

class Scenario : public subdomain {
public:
  using Serial = std::uint32_t;

  Scenario() : serialNumber_(nextSerial_()) {}
  ~Scenario() override = default;

  Scenario(const Scenario&) = delete;
  Scenario& operator=(const Scenario&) = delete;

  void setup(Config& cfg, const std::string& scenName);

  void performTask(void) override;

  // Also interrupts any solver call this scenario has in flight (best effort)
  void signalAbort(void) override;

  objAndBnd LagrangianSolve(void);

  const std::vector<double>& getX() const { return x; }

  void fetchXinto(std::vector<double>& x_) { x_ = x; }
  const std::vector<double>& getXExport() const { return xExport; }
  void fetchXExportInto(std::vector<double>& x_) { x_ = xExport; }
  void computeYinto(std::vector<double>& y);

  void inplaceUpdateW(const std::vector<double>& src) {
    assert(src.size() == W.size());
    std::copy(src.begin(), src.end(), W.begin());
  }

  void inplaceUpdateZ(const std::vector<double>& src) {
    assert(src.size() == z.size());
    std::copy(src.begin(), src.end(), z.begin());
  }

  int getTotalNonantCount() const { return problemSolver.getTotalNonantCount(); }
  // Subproblem objective at the exported point, as of the export decision.
  double getLastExportObjVal() const { return problemSolver.getLastExportObjVal(); }
  int getNodeNonantCount(const std::string& nodeName) const { return problemSolver.getNodeNonantCount(nodeName); }

  const NodeDataMap& getTreeNodeDataMap() const { return problemSolver.getTreeNodeDataMap(); }
  int getGlobalNodeCount() const noexcept { return problemSolver.getGlobalNodeCount(); }

  const std::vector<double>& getRho() const { return problemSolver.getRho(); }
  void setRho(const std::vector<double>& newRho) { problemSolver.updateRho(newRho); }

  const std::vector<double>& getOmicron() const { return problemSolver.getOmicron(); }
  void setOmicron(const std::vector<double>& newO) { problemSolver.setOmicron(newO); }
  bool omicronIsFrozen() const { return problemSolver.omicronIsFrozen(); }

  // Config accessors proxied from ProblemWithSolvers (so FWAPH can read without reaching into solvers)
  double getOmicronRhoMultiplier() const { return problemSolver.getOmicronRhoMultiplier(); }
  bool getCheckOmicronInvariant() const { return problemSolver.getCheckOmicronInvariant(); }

  void setNonantBounds(const std::vector<double>& lb, const std::vector<double>& ub) {
    problemSolver.updateNonantBounds(lb, ub);
  }
  const std::vector<double>& getNonantLowerBounds() const { return problemSolver.getNonantLowerBounds(); }
  const std::vector<double>& getNonantUpperBounds() const { return problemSolver.getNonantUpperBounds(); }

  double getProbability() const { return problemSolver.getProbability(); }
  const std::string& getProblemName() const { return problemSolver.getProblemName(); }

  bool hasNumericalBailout() const { return problemSolver.hasNumericalBailout(); }

  // True if the original objective for this scenario is quadratic
  bool hasQuadraticObjective() const { return problemSolver.hasQuadraticObjective(); }

  const objAndBnd& getLagrangianResults() const { return lagrangianResults; }

  double getLastFWSolveTime()  const { return lastFWSolveTime_; }
  double getLastFWNLTime()     const { return lastFWNLTime_; }
  double getLastFWMILPTime()   const { return lastFWMILPTime_; }
  int    getLastFWMILPSolves() const { return lastFWMILPSolves_; }

  std::vector<std::string>& getNodeNames() { return nodeNames; }
  std::vector<int>& getNodeSerialNumbers() { return nodeSerialNumbers; }

  const std::vector<int>& getNodeVarIndexes(const std::string& nodeName) const {
    return problemSolver.getNodeVarIndexes(nodeName);
  }

  int varIndexToNonantPos(int varIndex) const { return problemSolver.varIndexToNonantPos(varIndex); }

  void setW(const std::vector<double>& Win) { inplaceUpdateW(Win); }
  const std::vector<double>& getW() const { return W; }

  void xToCsv(const std::string& dirName) const;
  void WToCsv(const std::string& dirName) const;

  void _csvGuts(const std::string& dirName,
                const std::string& fileName,
                const std::vector<double>& vec) const;

  Serial getSerialNumber() const noexcept { return serialNumber_; }

  SolverInterface* getLagrangianPtr() { return problemSolver.getLagrangianPtr(); }
  const SolverInterface* getLagrangianPtr() const { return problemSolver.getLagrangianPtr(); }

protected:
  std::vector<std::string> nodeNames;
  std::vector<int> nodeSerialNumbers;

  void FWsolve(void);

  std::vector<double> W;
  std::vector<double> z;
  std::vector<double> x;        // proximal minimizer: the algorithm's iterate
  std::vector<double> xExport;  // a point of K_s: the exported candidate

  ProblemWithSolvers problemSolver;
  objAndBnd lagrangianResults;

  double lastFWSolveTime_ = 0.0;
  double lastFWNLTime_    = 0.0;
  double lastFWMILPTime_  = 0.0;
  int    lastFWMILPSolves_ = 0;

  bool setupWasCalled = false;

  static Serial nextSerial_() noexcept {
    static std::atomic<Serial> global{0};
    return global.fetch_add(1, std::memory_order_relaxed);
  }

  Serial serialNumber_;
};
