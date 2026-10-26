///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#pragma once

/*
  Interface to the lower level problem and solver code for one scenario.
  Notes:
  - Naming conventions differ between the code that calls this and the code this calls.
*/

#include "config/config.hpp"
#include "solver_interfaces/fw.hpp"
#include "utils/timer.hpp"
#include "utils/utilities.hpp"
#include "WTrace.hpp"
#include <optional>

// Exactly one of USE_CPLEX, USE_GUROBI, USE_XPRESS, USE_HIGHS must be defined
#if ( defined(USE_CPLEX) + defined(USE_GUROBI) + defined(USE_XPRESS) \
      + defined(USE_HIGHS) ) != 1
#error "Exactly one of USE_CPLEX, USE_GUROBI, USE_XPRESS or USE_HIGHS must be defined."
#endif


#ifdef USE_CPLEX
#include "solver_interfaces/cplex/cplex_interface.hpp"
#include "solver_interfaces/cplex/cplex_params.hpp"
#include "solver_interfaces/cplex/nl_cplex.hpp"
#include "solver_interfaces/cplex/grad_cplex.hpp"
#endif

#ifdef USE_GUROBI
#include "solver_interfaces/gurobi/gurobi_interface.hpp"
#include "solver_interfaces/gurobi/gurobi_params.hpp"
#include "solver_interfaces/gurobi/nl_gurobi.hpp"
#include "solver_interfaces/gurobi/grad_gurobi.hpp"
#endif

#ifdef USE_XPRESS
#include "solver_interfaces/xpress/xpress_interface.hpp"
#include "solver_interfaces/xpress/xpress_params.hpp"
#include "solver_interfaces/xpress/nl_xpress.hpp"
#include "solver_interfaces/xpress/grad_xpress.hpp"
#endif

#ifdef USE_HIGHS
#include "solver_interfaces/highs/highs_interface.hpp"
#include "solver_interfaces/highs/highs_params.hpp"
#include "solver_interfaces/highs/nl_highs.hpp"
#include "solver_interfaces/highs/grad_highs.hpp"
#endif


class ProblemWithSolvers : virtual public debugLevels {
public:
  ProblemWithSolvers() = default;
  ~ProblemWithSolvers();

  // setup needs to be called before you use this class
  void setup(Config& cfg,
        const std::string& probName,
        std::vector<double>& W_,
        std::vector<double>& z_);

  
  // aka scenario name (which can be confusing)
  const std::string& getProblemName() const {
    return problemName;
  }
 
  static void registerOptions(Config& config);

  const int getTotalNonantCount() const {return totalNonantCount; };
  const int getGlobalNodeCount() const {return globalNodeCount; };

  bool hasNumericalBailout() const {
    return (Lagrangian && Lagrangian->hasNumericalBailout())
        || (nl && nl->hasNumericalBailout())
        || (gradient && gradient->hasNumericalBailout());
  }

  // Two solve functions that put values into x
  objAndBnd doLagrangianSolve(std::vector<double>& x);
  // x receives the proximal minimizer (what the algorithm iterates on);
  // xExport receives a point of K_s (what the candidate-solution
  // consumers get).  They coincide only when FW-export-policy is prox.
  void doFWsolve(std::vector<double>& x, std::vector<double>& xExport);

  // Cooperative abort: attach the controller's abort flag to every solver
  // object (call after setup), and let the control thread interrupt solves
  // that are already running.  Both are no-ops when --abort-inflight-solves
  // is false.  Interrupted solve results must be discarded by the caller.
  void attachAbortFlag(const std::atomic<bool>* flag);
  void interruptInFlightSolves();

  // Timing accessors for the most recent FW solve
  double getLastFWNLTime()     const { return FW->getAccumulatedNLTime();   }
  double getLastFWMILPTime()   const { return FW->getAccumulatedMILPTime(); }
  int    getLastFWMILPSolves() const { return FW->getMILPSolvesSoFar();    }

  // Subproblem objective of the point the most recent FW solve exported, as
  // measured when the export was chosen.  NaN under --FW-export-policy prox.
  double getLastExportObjVal()  const { return FW->getLastExportObjVal();  }

  // True if this scenario's original objective has quadratic terms.  The
  // Lagrangian object holds the model as it was read from the file, so it is
  // the one to ask.
  bool hasQuadraticObjective() const {
    return Lagrangian && Lagrangian->hasOriginalQuadratic();
  }

  // All of the interfaces for this problem have, of course, the same map.
  const NodeDataMap& getTreeNodeDataMap() const {
    return Lagrangian->getTreeNodeDataMap();
  }

  // It is a little dangerous to mess with this
  // non-owning accessors (DO NOT return unique_ptr by value here)
  SolverInterface* getLagrangianPtr() {
    return Lagrangian.get();
  }

  const SolverInterface* getLagrangianPtr() const {
    return Lagrangian.get();
  }

  
  // Get the scenario probability from below (it came from the disk)
  const double getProbability() const {
    return Lagrangian->getProbability();
  }

  // set the value of rho and also update the solver objects  (remember that rho is special)
  void updateRho(const std::vector<double> &newRho);

  // get a read-only reference to the current rho
  const std::vector<double>& getRho() const noexcept { return rho; }

  // Omicron (dual-side metric weights). Static after freeze; see design/omicron_implementation_plan.md.
  const std::vector<double>& getOmicron() const noexcept { return omicron; }
  bool omicronIsFrozen() const noexcept { return omicronFrozen_; }
  // One-shot: bounds-check then install. Throws if already frozen or out of bounds.
  void setOmicron(const std::vector<double>& newO);

  // Config accessors (read by FWAPH to orchestrate the post-iter-0 freeze).
  double getOmicronRhoMultiplier() const noexcept { return omicronRhoMultiplier_; }
  double getOmicronMin() const noexcept { return omicronMin_; }
  double getOmicronMax() const noexcept { return omicronMax_; }
  bool getCheckOmicronInvariant() const noexcept { return checkOmicronInvariant_; }

  // Variable bounds on nonant variables (reduced-costs bridge).
  // NL is deliberately skipped — see design/reduced-costs-bridge.md.
  void updateNonantBounds(const std::vector<double>& lb, const std::vector<double>& ub);
  const std::vector<double>& getNonantLowerBounds() const { return Lagrangian->getNonantLowerBounds(); }
  const std::vector<double>& getNonantUpperBounds() const { return Lagrangian->getNonantUpperBounds(); }

  // this function should be called only for output, so speed is not important
  std::vector<std::string> getNonantNames() const {
    return Lagrangian->nonantNamesVector();
  }

  // passing nonant data from below to above
  int getNodeNonantCount(const std::string& nodeName) const {
    return Lagrangian->getNodeNonantCount(nodeName);
  }
  // expose node->var-index list from the Lagrangian interface
  const std::vector<int>& getNodeVarIndexes(const std::string& nodeName) const {
    return Lagrangian->getNodeIndexes(nodeName);
  }

  // map solver var-index -> dense nonant index (nai)
  int varIndexToNonantPos(int varIndex) const {
    auto nai = Lagrangian->iToNai(varIndex);
    return static_cast<int>(nai);
  }  

  // this is for testing (it is no longer the Lagrangian after this)
  void enableLagrangianProxAt(std::vector<double>& z);

  // to output an mps (or lp) file
  void writeLagrangianModel(const std::string& filename) const;

  // Install W directly and update the Lagrangian objective
  void setWFromTrace(const std::vector<double>& W);



protected:
#ifdef USE_CPLEX
  std::unique_ptr<CPLEXInterface> Lagrangian;
  std::unique_ptr<CPLEXNL> nl;
  std::unique_ptr<CPLEXGrad> gradient;
  std::unique_ptr<FrankWolfe<CPLEXNL, CPLEXGrad>> FW;
#endif
#ifdef USE_GUROBI
  std::unique_ptr<GurobiInterface> Lagrangian;
  std::unique_ptr<GurobiNL> nl;
  std::unique_ptr<GurobiGrad> gradient;
  std::unique_ptr<FrankWolfe<GurobiNL, GurobiGrad>> FW;
#endif
#ifdef USE_XPRESS
  std::unique_ptr<XpressInterface> Lagrangian;
  std::unique_ptr<XpressNL> nl;
  std::unique_ptr<XpressGrad> gradient;
  std::unique_ptr<FrankWolfe<XpressNL, XpressGrad>> FW;
#endif
#ifdef USE_HIGHS
  std::unique_ptr<HiGHSInterface> Lagrangian;
  std::unique_ptr<HiGHSNL> nl;
  std::unique_ptr<HiGHSGrad> gradient;
  std::unique_ptr<FrankWolfe<HiGHSNL, HiGHSGrad>> FW;
#endif
  // solver params defined for base class and instantiated later based on compiler directive
  std::unique_ptr<SolverParams> iter0Params;
  std::unique_ptr<SolverParams> nlParams;
  std::unique_ptr<SolverParams> gradientParams;
  std::unique_ptr<SolverParams> FWParams;

  // these strings are all paths
  std::string problemName;
  std::string rhoCsv;
  std::string iter0Json;
  std::string nlJson;
  std::string gradJson;
  std::string logFileName;

  std::vector<double> rhoAsRead; // from an input file
  std::vector<double> rho;       // current rho vector
  std::optional<double> rhoAll; // if set, override rho-csv and use a scalar for all nonants

  // Omicron state (see design/omicron_implementation_plan.md).
  // Initialized to all-ones at setup. Frozen exactly once: either at setup
  // (when --omicron-all or --omicron-csv is given) or post-iter-0 via
  // FWAPH::freezeOmicron (default: multiplier * rho at that moment).
  std::vector<double> omicron;
  bool omicronFrozen_ = false;
  std::string omicronCsv_;
  std::optional<double> omicronAll_;
  double omicronRhoMultiplier_ = 1.0;
  double omicronMin_ = 1e-6;
  double omicronMax_ = 1e6;
  bool checkOmicronInvariant_ = true;


  // local functions
  void _getFromCfg(Config &cfg);
  void _readRho();
  void _readOmicron(std::vector<double>& out);
  void _createInterfaceObjects(Config& cfg);
  void _initRho();
  

  // expose some problem data from below
  int totalNonantCount;
  int globalNodeCount;

  // Multiplier and "center" data owned by the caller
  // Non-owning bindings to caller-owned containers
  std::vector<double>* W_ = nullptr;
  std::vector<double>* z_ = nullptr;

  // Cooperative abort state (see attachAbortFlag)
  const std::atomic<bool>* abortFlagPtr_ = nullptr;
  bool abortInFlightEnabled_ = true;

  // A full x vector, presumably from Lagrangian
  std::vector <double> fullX;

  // Algorithm parameters
  int tMax;
  double tau;

  // to make sure setup is called before use
  bool setupWasCalled = false;

  // To create traces for papers and talks
  std::unique_ptr<WTrace> wTrace_;
};
