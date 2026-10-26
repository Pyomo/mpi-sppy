#ifndef CPLEX_INTERFACE_HPP
#define CPLEX_INTERFACE_HPP

// TBD List:
//    - No support for models with quadratic objective; need to detect that.
//    - Grr: the sizes example needs stage cost Expressions (only two r's...)
//    - Maybe put the maps and vars in a struct to get them all at once
//    - related: Improve loop efficiency, particularly the map building loops
//    - related: maybe standardize loop idioms a little more
//    - Think harder about references versus copies of var names in particular
// Notes for users:
//     - See naming.txt in the parent directory
//     - We expect the caller to use references to the W vector etc. to update it
//       so the caller needs to keep track of whether the model needs to be
//       solved.
//     - Important: the caller owns W and z. We handle rho differently.
//     - If the caller wants a squared term, the caller needs to call the add
//       function and also set rho.
// Notes for developers and users:
//     - Whenever we need solution values, we do cplex.getValues(vals, vars);
//       to be sure the values are fresh. (In normal operation, that will be
//       done only once per iteration of the caller.)

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

The values for dense vectors W, rho, and z are supplied by the caller. A
reference to W and z vectors should be obtained from the caller and then
the caller can change the values. To send the new values to the solver,
the caller call updateProxObjective(). The values for rho need
to be swapped out using a setter call.

*/

/* Notes about CPLEX and quadradatic objective function terms:
You cannot simply change the coefficient of a quadratic term like you
can for a linear term. You need to subtract the term with the old
coefficient and then add the new one. 

So rho will be handled in a manor that is different from W, z, because
we need to easily detect that it might have changed.

BTW:
min y
s.t. y = x*x
will not work because cplex says the constraint is non-convex.
 */

#include <ilcplex/ilocplex.h>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <optional>
#include <unordered_set>
#include <fstream>   // new: for persistent logfile stream
#include <memory>    // new: for unique_ptr
#include "../solver_interface.hpp"
#include "../json.hpp"
#include "../../utils/utils.hpp"

using NonantTreeNameMap = std::map<std::string, std::vector<std::string>>;
using NonantTreeIndexMap = std::map<std::string, std::vector<int>>;


//===============================================

class CPLEXInterface : public SolverInterface {
public:
  // Existing constructor (no logfile) delegates to the two-arg version in the .cpp
  explicit CPLEXInterface(const std::string& scenarioName);
  // New overload: scenario name plus log file name to which CPLEX output will be redirected.
  CPLEXInterface(const std::string& scenarioName, const std::string& logFileName);

  virtual ~CPLEXInterface() override;
  // can be moved, but not copied
  CPLEXInterface(const CPLEXInterface&) = delete;
  CPLEXInterface& operator=(const CPLEXInterface&) = delete;
  CPLEXInterface(CPLEXInterface&&) = default;
  CPLEXInterface& operator=(CPLEXInterface&&) = default;

  objAndBnd solve() override;
  void interruptSolve() override;
  std::string getVarName(const size_t i) const override {
    return vars[i].getName();
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

  // Set solve parameters (need two so we can match the cplex signatures)
  void setCplexParam(const IloCplex::NumParam param, const double value) {
    try {
      cplex.setParam(param, value);
    }
    catch (IloException& e) {
      std::cerr << "Error Message: " << e.getMessage() << std::endl;
      e.end();  // Properly clean up the exception object
    } catch (...) {
      std::cerr << "Unknown exception caught!" << std::endl;
    }
  };
  void setCplexParam(const IloCplex::IntParam param, const int value) {
    cplex.setParam(param, value);
  };

  // Algorithmic data is all declared at the parent level
  
  // from here down are functions the caller should rarely, if ever, need.
  void printSolution() override;
  void nonantSolutionVectorToCsv(std::string fileName) const override;
  void exportModel(const std::string fname) override {  // have an mps (or lp) file name suffix
    cplex.exportModel(fname.c_str());
  };
  bool setStartAtCurrent() override; 
  void setLogFile(const std::string fileName) override;
   
protected:
  // cplex model objects
  IloEnv env;
  IloModel model;
  IloCplex cplex;
  IloNumVarArray vars;
  IloRangeArray constraints;
  IloObjective objective;

  // For cross-thread solve interruption; abort() persists until the clear()
  // at the top of solve(), so stray requests cannot leak into a later solve.
  IloCplex::Aborter aborter_;

  void _buildNonantsMaps();
  
  // let a child class (e.g. NL) start over
  void clearModel() override;

  // Optional log file name and stream used to capture CPLEX textual output.
  // If logFileName_ is empty, no redirection is performed.
  std::unique_ptr<std::ofstream> logStream_; // keep the stream alive for the lifetime of cplex
};
#endif // CPLEX_INTERFACE_HPP
