///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "spControllerBase.hpp"
#include "utils/utilities.hpp"
#include "utils/utils.hpp"
#include "WTrace.hpp"
#include "SolveTimesTrace.hpp"

#include <atomic>
#include <fstream>
#include <limits>

// Controller class for FW-APH

class FWAPH :
public spControllerBase
{
public:

  // Constructor

  FWAPH() {
    subdomainName = "scenario";
    numTaskTypes = 2;
    taskName.resize(2);
    taskName[0] = "Lagrangian solve";
    taskName[1] = "FW solve";
  };


  // Setup

  void setup(Config& cfg) override;

  // Register Options

  static void registerOptions(Config& cfg);

  // Print Results

  void printResults();

  // the most recently computed bound (which might not be all that recent)
  double getCurrentLagrangianBound() {
    return currentLagrangianBound;
  };

  // for the optional W trace file
  void saveWTrace();

  // Function for debugging: reports how far W is from the nonanticipative
  // complement, per node.  COLLECTIVE on the node communicators, so every rank
  // must call it; it gates itself on verbosity >= 4.
  void diagCheckWInNPerp(const std::string& tag);

  // --- Stepping API (for mpi-sppy integration) ---

  // Run exactly n cycles then return control to caller.
  // Returns the number of cycles actually completed (may be less if
  // convergence or external termination was detected).
  int runNCycles(int n);

  // Current cycle count
  int getCurrentCycle() const { return cycleCount; }

  // Whether the algorithm has converged (internal convergence or external)
  bool isConverged() const { return converged_; }

  // --- External termination (for mpi-sppy hub mode) ---

  // Set a pointer to an external terminate flag. When *flag becomes nonzero,
  // the event loop will exit at the next safe point (between comm phases).
  void setExternalTerminateFlag(std::atomic<int>* flag) {
    externalTerminateFlag_ = flag;
  }

  // --- Hub mode ---

  // When true, skip the final Lagrangian bound computation in terminationSequence
  void setHubMode(bool on) { hubMode_ = on; }
  bool getHubMode() const { return hubMode_; }

  // Run a Lagrangian solve on every local scenario at the current W and
  // reduce to the expected bound, which is returned (and also becomes the
  // value reported by getCurrentLagrangianBound()).  This is the body of
  // terminationSequence(); hub mode skips that, so the mpi-sppy driver calls
  // this directly when --aphfw-final-lagrangian is set.  Collective: every
  // rank in comm must call it.
  double computeFinalLagrangianBound();

  // --- Data export (flat arrays for mpi-sppy send buffers) ---

  // Fill buf with all local scenarios' nonant (x) values, concatenated
  // in local scenario order. len must equal sum of getTotalNonantCount()
  // across local scenarios.
  void getAllLocalNonants(double* buf, int len) const;
  // The algorithm's own iterate, and the probability-weighted mean of it.
  // getAllLocalNonants exports a candidate solution instead; see
  // design/fw-export-split.md.
  void getAllLocalProxX(double* buf, int len) const;
  void getAllLocalXmean(double* buf, int len) const;

  // Fill buf with all local scenarios' W values, same layout as above.
  void getAllLocalW(double* buf, int len) const;

  // Fill buf with all local scenarios' xbar (z) values, same layout as above.
  void getAllLocalXbar(double* buf, int len) const;

  // Fill buf with all local scenarios' rho values, same layout as above.
  void getAllLocalRho(double* buf, int len) const;

  // Update rho for all local scenarios from buf (same layout as above).
  // Triggers objective rebuild on each affected scenario's next solve.
  void setAllLocalRho(const double* buf, int len);

  // Fill buf with all local scenarios' omicron values, same layout as rho.
  void getAllLocalOmicron(double* buf, int len) const;

  // One-shot: freeze omicron = multiplier * current rho across all local scenarios.
  // Throws if any scenario's omicron is already frozen. Runs S(N)-constancy check
  // across local scenarios if --check-omicron-invariant is true.
  void freezeOmicron(double multiplier);

  // Has omicron been frozen on every local scenario?
  bool omicronIsFrozen() const;

  // Fill lbBuf/ubBuf with all local scenarios' nonant bounds (same layout).
  void getAllLocalNonantBounds(double* lbBuf, double* ubBuf, int len) const;

  // Update nonant bounds for all local scenarios (same layout).
  // Applied to Lagrangian and Gradient solvers; NL is deliberately skipped.
  void setAllLocalNonantBounds(const double* lbBuf, const double* ubBuf, int len);

  // Return the scenario names owned by this rank, in local order
  std::vector<std::string> getLocalScenarioNames() const;

  // Return the total length of the flat nonant vector for this rank
  int getLocalNonantLength() const;

  // Destructor

  ~FWAPH() { };
  

protected:

  // Keywords for communication phases

  enum commPhases { leadingSyncPhase, vectorReductionPhase, scalarReductionPhase};


  // Start up sequence

  void startupSequence() override;


  // Termination sequence

  void terminationSequence() override;


  // Override to support stepping mode (don't abort tasks when cycle target reached)
  void onMainLoopTerminate() override;


  // What happens when a round of communication ends

  bool handleCompletedCommunication() override;


  // Application-specific conditions allowing dispatch of a scenario

  bool dispatchAllowed() override;


  // Decide which scenario to start working on next

  std::pair<std::set<int>::iterator,int> bestTaskToDispatch(void) override;


  // To take appropriate action after a scenario finishes solving

  void handleCompletedTask(int s) override;


  // Start a new communication cycle

  void startCommCycle(void) override;


  // Enhance startTask so it keeps track of last dispatch time for FW solves

  void startTask(int s, int taskType) override;


  // Start a leading sync (first, possibly repeating part of the communication
  // cycle)

  void launchLeadingSync(void);

  // Start the reduction for a node
  void launchNodeReduction(ScenarioTreeNode&);

  // Start the reduction of scalars for a node
  void launchScalarReduction(ScenarioTreeNode&);

  // Figure out if termination conditions are met

  bool convergenceDetected() {
    if (displayProgress && myRank == 0) {
      std::cout << "displayProgress: normU, primalTermThresh; normV, dualTermThresh: ";
      std::cout << normU << ", " << primalTermThresh << "; " << normV << ", " << dualTermThresh << std::endl;
    }
    return normU <= primalTermThresh && normV <= dualTermThresh; 
  };

  
  // Compute nodal values for u, v and put them in the right places.
  void computeUVforNode(ScenarioTreeNode&);


  // Do the calculations related to projection (the end of the algorithm).
  // Returns true if tau fell below tolerance and the run should terminate.
  bool projectionStep();

  // Transfer u and v ?xxxx
  void setUV(ScenarioTreeNode& node);
  
  //// Parameters that can be set (maybe indirectly) from the command line

  double primalTermThresh;     // Primal termination tolerance (applied to u)
  double dualTermThresh;       // Dual termination tolerance (applied to v)
  double tauTolerance;         // "emergency" termination because tau is too small
  double gammaInverse;         // primal/dual scaling (applied to norm v)
  double nu;                   // overrelaxation parameter (TBD: dynamic)
  int    stalenessThreshhold;  // Max to wait between dispatches of a scenario (M)
  bool   leadingDispatchOnly;  // Don't dispatch while a projection is in progress
  bool   blockSync;            // Allow only block asynchronism
  bool   displayProgress;      // output convergence information at every "iteration"
  int    aphMaxCycles;         // A form of max iterations
  int    cycleTrigger;         // Number of solved scenarios that trigger a projection
  int    localTrigger;         // Local number of scenarios needed for a projection
                               //   (applies only in blockSync mode)
  bool   iter0FWStep;          // Finish startup with a synchronous FW step on every
                               //   scenario (needed when the objective is nonlinear)
  double iter0WMultiplier;     // Startup W = m * rho * (xbase - z0); 0 gives W = 0

  bool numericalBailoutDetected = false;  // set when any local scenario hits unrecoverable suboptimal

  std::string csvOutputDirName = "";  // Directory for x and W files
  

  //// Data members

  // List of scenarios queued for "projection" or currently in projection

  std::list<int> queuedForProjection;
  std::list<int> inProjection;

  // How many sycn reductions were done waiting to start projections
  // (counts syncs to count finished scenarios)

  int syncCount;

  // Counts of scenarios queued for reduction

  int localQueuedCount;       // On this rank
  int commQueuedCount;        // Buffer to hold localQueuedCount when reducing
  int globalQueuedCount;      // Across all ranks

  

  // vectors of scenario vectors; they are owned here, and given sizes in setup()
  // values are computed in this object; values are copied to the Scenario object
  // (Scenario owns its own copy)
  
  std::vector<std::vector<double>> W;   // Lagrange multipliers of nonant constraints
  std::vector<std::vector<double>> x;   // Primal elements of graph-of-operator pairs (obtained from solves)
  std::vector<std::vector<double>> xExport;  // Candidate solutions (points of K_s) for the spokes
  std::vector<std::vector<double>> y;   // Dual elements of graph-of-operator pairs
  std::vector<std::vector<double>> u;   // To compute projections

  // (two vectors, z and v, that could be stored node-wise, but aren't right now)
  
  std::vector<std::vector<double>> z;   // Average primal value (xbar in PH)
  std::vector<std::vector<double>> v;   // To compute projections

  std::vector<double> phi;              // By-scenario hyperplane violations

  std::vector<int> lastDispatchCycle;   // The projection counter the last time
                                        // each scenario was dispatched.

  double normU;                         // Norms of u and v
  double normV;

  // scalar useful for data processing
  int largestNodeNonantSize;

  /* removed Jan 31, 2026 replaced with nodal buffers 
  // buffers for reductions  (these need to be 2*largestNodeNonantSize) 
  std::vector<double> nodeReduceInBuf;  // local avg xy for reduction
  std::vector<double> nodeReduceOutBuf; // global avg xy from reduction
  */

  // buffers for scalar reductions; [0]=phi, [1]=sqNormU, [2]=sqNormV, [3]=numericalBailout
  // [0] phi sum, [1] unweighted ||u||^2 (for termination check),
  // [2] ||v||^2, [3] numerical bailout flag, [4] omicron-weighted ||u||^2_Omega.
  std::array<double, 5> scalarReduceInBuf{};   // zeros
  std::array<double, 5> scalarReduceOutBuf{};  // zeros

  // optional per-cycle oscillation trace (rank 0 only)
  std::string oscTracePrefix_;
  std::ofstream oscTraceFile_;

  // raison d'etre.  -infinity until startupSequence() computes the iter-0
  // bound: getCurrentLagrangianBound() is reachable before then (the stepping
  // API defers startup to the first runNCycles call), and a bound that has not
  // been computed yet must read as "no bound", not as uninitialized memory.
  double currentLagrangianBound = -std::numeric_limits<double>::infinity();

  // --- Stepping / hub mode state ---
  bool hubMode_ = false;
  bool converged_ = false;
  bool startupDone_ = false;      // whether startupSequence has been called
  int cycleTarget_ = 0;           // target cycle count for runNCycles
  bool maxCyclesReached() const;
  std::atomic<int>* externalTerminateFlag_ = nullptr;

  // local functions
  void _computeCurrentLagrangianBound();

  // Collective over each node's communicator; aborts if the startup W is not in
  // the orthogonal complement of the nonanticipative subspace.  Pass the same
  // serial-number-sorted node vector used for the other per-node collectives.
  void _checkWOrthogonality(const std::vector<ScenarioTreeNode*>& nodes);

  // Reduce one node's probability-weighted sum of W (which must vanish for W to
  // be in the complement) and the matching sum of magnitudes.  Collective on the
  // node communicator.  Shared by _checkWOrthogonality and diagCheckWInNPerp so
  // the two cannot drift apart in weighting or tolerance basis.
  void _reduceNodeWResidual(ScenarioTreeNode& node,
                            std::vector<double>& residual,
                            std::vector<double>& scale);

  // Local nodes sorted by serial number, so every rank issues its per-node
  // blocking collectives in a globally consistent order.
  std::vector<ScenarioTreeNode*> _nodesInCollectiveOrder();

  // for optional file with a W trace
  std::unique_ptr<WTrace> wTrace_;  // one per rank
  std::string wTracePrefix_;        // optional config option name you chose

  // for optional per-rank FW solve timing trace
  std::unique_ptr<SolveTimesTrace> solveTimesTrace_;
  std::string solveTimesTracePrefix_;
};

