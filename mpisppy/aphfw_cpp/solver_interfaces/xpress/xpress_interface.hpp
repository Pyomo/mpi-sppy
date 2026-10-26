#ifndef XPRESS_INTERFACE_HPP
#define XPRESS_INTERFACE_HPP

/* Xpress solver interface using the Xpress C API (xprs.h).
 *
 * Key differences from Gurobi/CPLEX:
 *   - Variables are accessed by index, not by objects.
 *   - Solution values are stored in a local vector after each solve.
 *   - Quadratic objective uses the 0.5 * x^T Q x convention:
 *       to represent coeff * x_i^2 in the objective, set Q[i][i] = 2*coeff.
 *   - XPRSinit() must be called once globally before any XPRScreateprob().
 */

#include <xprs.h>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <optional>
#include <unordered_set>
#include <mutex>
#include "../solver_interface.hpp"
#include "../json.hpp"
#include "../../utils/utils.hpp"

using NonantTreeNameMap = std::map<std::string, std::vector<std::string>>;
using NonantTreeIndexMap = std::map<std::string, std::vector<int>>;

//===============================================

class XpressInterface : public SolverInterface {
public:
  XpressInterface(const std::string& scenarioName, const std::string& logFileName);
  virtual ~XpressInterface() override;

  // move-only
  XpressInterface(const XpressInterface&) = delete;
  XpressInterface& operator=(const XpressInterface&) = delete;
  XpressInterface(XpressInterface&&) = default;
  XpressInterface& operator=(XpressInterface&&) = default;

  objAndBnd solve() override;
  void interruptSolve() override;
  std::string getVarName(const size_t i) const override {
    return varNames_[i];
  }
  virtual std::vector<double> copyNonantSolution() const override;
  virtual std::vector<double> copyEntireSolution() const override;
  virtual void extractNonantSolution(std::vector<double>& nonantSolution) const override;
  virtual void extractNodeSolution(const std::string nodeName,
                                   std::vector<double>& nodeSolution) const override;
  virtual void fullObjectiveRebuild() override;
  void applyBoundsToSolver() override;
  double getVarLB(int varIdx) const override;
  double getVarUB(int varIdx) const override;

  void setLogFile(const std::string fileName) override;
  void logMsg(std::string msg) const override {
    std::cout << "Xpress log message to cout: " << msg << std::endl;
  }

  // Xpress parameter setters (overloaded for int and double)
  void setXpressIntControl(int control, int value);
  void setXpressDblControl(int control, double value);

  void printSolution() override;
  void nonantSolutionVectorToCsv(std::string fileName) const override;
  void exportModel(const std::string fname) override;
  bool setStartAtCurrent() override;

protected:
  // Xpress problem handle
  XPRSprob prob_ = nullptr;
  int numVars = 0;
  bool isMIP_ = false;

  // Variable names (indexed by column index)
  std::vector<std::string> varNames_;

  // Most-recent solution (updated after each solve)
  mutable std::vector<double> solution_;

  void _buildNonantsMaps();
  void clearModel() override;

  // Helper: check Xpress return code, throw on error
  void xprsCheck(int ret, const char* context) const;

private:
  // One-time global Xpress library initialization
  static std::once_flag s_initFlag_;
  static void globalInit();

  // Helper to load variable names from the problem
  void loadVarNames();

  // Helper to refresh solution_ from solver after a solve
  void fetchSolution() const;
};

#endif // XPRESS_INTERFACE_HPP
