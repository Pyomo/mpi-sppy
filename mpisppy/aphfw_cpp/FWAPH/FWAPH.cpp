///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "FWAPH.hpp"
#include "ProblemWithSolvers.hpp"
#include "utils/timer.hpp"
#include <cmath>
#include <array>
#include <iomanip>
#include <limits>

static void dbg_check_node_uv(int myRank,
                              int verbosity,
                              int level,
                              const ScenarioTreeNode& node,
                              const std::vector<double>& scenProb,
                              const std::vector<std::vector<double>>& u,
                              const std::vector<std::vector<double>>& v,
                              const std::vector<std::vector<double>>& y,
                              double nodeUncondProb)
{
  if (myRank != 0 || verbosity < level) return;

  const auto& scenList = node.getNodeLocalScenarioIndexList(); // needs const overload (see below)
  if (scenList.empty()) return;

  int naStart = node.getDenseNonantStart();
  int xLen    = node.getDenseNonantLen();

  // Check only first couple entries to keep output short
  int lim = std::min(2, xLen);

  for (int j = 0; j < lim; ++j) {
    // 1) weighted sum pi*u should be ~0 on this node segment
    double sum_pi_u = 0.0;

    // 2) v should be constant and equal to ybar; compare v vs ybar
    double ybar = 0.0;

    // compute ybar
    for (int s : scenList) {
      double w = scenProb[s] / nodeUncondProb; // conditional prob weight within node
      ybar += w * y[s][naStart + j];
    }
    // compute sum pi*u and also check v consistency
    double max_v_dev = 0.0;
    for (int s : scenList) {
      double w = scenProb[s] / nodeUncondProb;
      sum_pi_u += w * u[s][naStart + j];
      double dev = std::abs(v[s][naStart + j] - ybar);
      if (dev > max_v_dev) max_v_dev = dev;
    }

    std::cerr << std::setprecision(17)
              << "[DIAG-UV] Node=" << node.getName()
              << " j=" << j
              << " sum_pi_u=" << sum_pi_u
              << " ybar=" << ybar
              << " max|v-ybar|=" << max_v_dev
              << "\n";
  }
}


void FWAPH::registerOptions(Config& config) { // static
  
  ProblemWithSolvers::registerOptions(config);
  
  config.addOption<double>("primal-term-thresh", 0.01, "Primal termination tolerance (applied to u)");
  config.addOption<double>("dual-term-thresh", 0.01, "Dual termination tolerance (applied to v)");
  config.addOption<double>("gamma-inverse", 1, "primal/dual scaling (applied to norm v)");
  config.addOption<double>("nu", 1, "APH overrelaxation parameter between 0 and 2 ");
  config.addOption<double>("APH-tau-tolerance", 1e-6, "If tau is below this value, 'emergency' terminate");
  config.addOption<int>("staleness-threshhold", 20, "Max to wait between dispatches of a scenario (M)");
  config.addOption<bool>("leading-dispatch-only", false, "Only dispatch scenarios during leading sync phase, not reduction");
  config.addOption<bool>("block-sync", false, "Allow only block asynchronism");
  config.addOption<bool>("display-progress", false, "results in convergence output at every iteration");
  config.addOption<int>("aph-max-cycles", 0, "Max APH 'iterations'");
  config.addOption<double>("solve-fraction", 1, "Fraction of solves needed before starting a cycle of computation");
  config.addOption<std::string>("csv-output-directory", "", "If provided, final x and W value files will go to this directory");
  config.addOption<std::string>("W-trace-prefix", "", " If given, there will be a W trace file whose name has this prefix");
  config.addOption<std::string>("solve-times-trace-prefix", "", "If given, one CSV per rank with per-FW-solve timing");
  config.addOption<std::string>("oscillation-trace-prefix", "", "If given, write per-cycle CSV with u,v,x,z,W,y for scenario 0 on rank 0");
  config.addOption<bool>("check-tree", false, "Causes in an input data check that should not be used with bundles createdy mpisppy");
  config.addOption<std::string>("iter0-fw-step", "auto",
    "Take a synchronous FW step at the end of startup so (x0,y0) is a valid pair: "
    "'auto' (only when the original objective is quadratic), 'on', or 'off'");
  config.addOptionNonNegative<double>("iter0-w-multiplier", 0.0,
    "Startup W = m * rho * (xbase - z0) with this m (0.0 gives W=0, the default)");

}

// setup does the work of a ctor
void FWAPH::setup(Config& cfg) {
  spControllerBase::setup(cfg); // sets up local scenarios and tree nodes
// FWAPH.cpp (inside FWAPH::setup)

  syncCount = 0;

  // queued-counts must start at 0 (otherwise launchLeadingSync reads garbage)
  localQueuedCount  = 0;
  commQueuedCount   = 0;
  globalQueuedCount = 0;

  // use the local scenarios to give the vectors their sizes
  int numScens = localScenarioIndexList.size();  // typing aid
  W.resize(numScens); 
  x.resize(numScens);
  xExport.resize(numScens);
  y.resize(numScens);
  u.resize(numScens);
  z.resize(numScens);
  v.resize(numScens);
  phi.resize(numScens);
  lastDispatchCycle.resize(numScens);
  for (int s = 0; s < numScens; s++) {
    int xSize = scenario(s).getTotalNonantCount();
    W[s].resize(xSize);
    x[s].resize(xSize);
    xExport[s].resize(xSize);
    y[s].resize(xSize);
    u[s].resize(xSize);
    z[s].resize(xSize);
    v[s].resize(xSize);
  }

  // now get the paramters from cfg
  primalTermThresh = cfg.getOptionValue<double>("primal-term-thresh");
  dualTermThresh  = cfg.getOptionValue<double>("dual-term-thresh");
  tauTolerance  = cfg.getOptionValue<double>("APH-tau-tolerance");
  gammaInverse  = cfg.getOptionValue<double>("gamma-inverse");
  nu = cfg.getOptionValue<double>("nu");
  stalenessThreshhold = cfg.getOptionValue<int>("staleness-threshhold");
  leadingDispatchOnly = cfg.getOptionValue<bool>("leading-dispatch-only");
  blockSync = cfg.getOptionValue<bool>("block-sync");
  displayProgress  = cfg.getOptionValue<bool>("display-progress");
  aphMaxCycles  = cfg.getOptionValue<int>("aph-max-cycles");
  iter0WMultiplier = cfg.getOptionValue<double>("iter0-w-multiplier");

  // Decide now whether startup ends with a synchronous FW step (see
  // startupSequence).  "auto" asks the scenarios whether the original
  // objective is quadratic; the answer has to be the same on every rank, so
  // reduce it.
  {
    const std::string mode = cfg.getOptionValue<std::string>("iter0-fw-step");
    if (mode == "on")
      iter0FWStep = true;
    else if (mode == "off")
      iter0FWStep = false;
    else if (mode == "auto") {
      int localQuad = 0;
      for (int s = 0; s < numScens; ++s)
        if (scenario(s).hasQuadraticObjective()) { localQuad = 1; break; }
      int globalQuad = 0;
      MPI_Allreduce(&localQuad, &globalQuad, 1, MPI_INT, MPI_MAX, comm);
      iter0FWStep = (globalQuad != 0);
    }
    else
      throw std::runtime_error("--iter0-fw-step must be 'auto', 'on', or 'off', not '"
                               + mode + "'");
  }

  // Continuing, set up cycle trigger

  double solveFraction = cfg.getOptionValue<double>("solve-fraction");

  if (blockSync)
  {
    // In block-sync situation, everybody must solve a specific fraction of
    // their local scenarios, and the global trigger is everybody being done
    localTrigger = std::ceil(numScens*solveFraction);
    MPI_Allreduce(&localTrigger, &cycleTrigger, 1, MPI_INT, MPI_SUM, comm);
  }
  else 
    cycleTrigger = std::round(globalScenCount*solveFraction);

  cycleTrigger = std::min(std::max(1, cycleTrigger), globalScenCount);

  // And also 

  csvOutputDirName = cfg.getOptionValue<std::string>("csv-output-directory");

  // If an output directory is specified, make sure it is empty
  // Note: there will be a barrier befor this is used, so we can safely deal with it here
  if ((csvOutputDirName != "") and (myRank == 0)) {
    ensure_fresh_output_dir(csvOutputDirName);
  }

  // This can't possibly fail, but...
  if (numSubdomains != (int)localScenarioIndexList.size()) {
    std::cerr << "FATAL: numSubdomains mismatch: " << numSubdomains
              << " vs localScenarioIndexList.size()=" << localScenarioIndexList.size() << "\n";
    MPI_Abort(comm, 1);
  }
  
  // optional W trace
  wTracePrefix_ = cfg.getOptionValue<std::string>("W-trace-prefix");

    if (!wTracePrefix_.empty()) {
        // One file per rank: e.g. WTrace_rank0.json
        std::string fname = wTracePrefix_ + "_rank" + std::to_string(getRank()) + ".json";
        wTrace_ = std::make_unique<WTrace>(fname);
    }

  // optional solve-times trace
  solveTimesTracePrefix_ = cfg.getOptionValue<std::string>("solve-times-trace-prefix");
  if (!solveTimesTracePrefix_.empty()) {
    std::string fname = solveTimesTracePrefix_ + "_rank" + std::to_string(getRank()) + ".csv";
    solveTimesTrace_ = std::make_unique<SolveTimesTrace>(fname);
  }

  // optional oscillation trace (rank 0 only)
  oscTracePrefix_ = cfg.getOptionValue<std::string>("oscillation-trace-prefix");
  if (!oscTracePrefix_.empty() && myRank == 0 && numSubdomains > 0) {
    std::string fname = oscTracePrefix_ + ".csv";
    oscTraceFile_.open(fname);
    if (!oscTraceFile_.is_open()) {
      std::cerr << "Warning: could not open oscillation trace file: " << fname << std::endl;
    } else {
      // Write header: cycle-level scalars + per-nonant columns for scenario 0
      oscTraceFile_ << "cycle,theta,phi,tau,tauOmega,normU,normV";
      int nNonants = static_cast<int>(x[0].size());
      for (int i = 0; i < nNonants; ++i) {
        oscTraceFile_ << ",x_" << i << ",z_" << i << ",u_" << i
                       << ",W_" << i << ",y_" << i << ",v_" << i
                       << ",o_" << i;
      }
      oscTraceFile_ << "\n";
    }
  }
}


// When a task completes:  if we're in the main event loop, move the scenario
// to the queue for being projected

void FWAPH::handleCompletedTask(int s)
{
  if (scenario(s).hasNumericalBailout()) {
    numericalBailoutDetected = true;
  }

  if (scenario(s).currentTaskType() == FWsolveType)
    {
      if (!terminateFlag)
      {
        localQueuedCount++;
        queuedForProjection.push_back(s);
      }
      if (solveTimesTrace_) {
        solveTimesTrace_->append(
          global_seconds(),
          lastDispatchCycle[s],
          scenario(s).getProblemName(),
          scenario(s).getLastFWSolveTime(),
          scenario(s).getLastFWNLTime(),
          scenario(s).getLastFWMILPTime(),
          scenario(s).getLastFWMILPSolves()
        );
      }
    }
}


// To handle completed communications.  Returns true when termination has been detected.

bool FWAPH::handleCompletedCommunication(void)
{
  DEBUGPR(1, rankTag << "FWAPH::handleCompletedCommunication, commPhase="
    << commPhase << ", active = " << setToString(active)
    << ", eligible = " << setToString(eligible));

  // We decide what to with the incoming message based on commPhase,
  // then set it to determine what to do next.

  switch (commPhase)
  {
  case leadingSyncPhase:                   // Leading sync done, ready to start reducing
    syncCount++;

    DEBUGPR(2, rankTag << "globalQueuedCount=" << globalQueuedCount);

    // globalQueuedCount should have been set by this communication.  If there
    // have been enough completed solves to trigger a projection, set commPhase
    // to 1 to signal starting a projection.  Otherwise it will remain zero
    // and start another leading sync.

    if (globalQueuedCount >= cycleTrigger)
      setCommPhase(vectorReductionPhase);
 
    break;

  case vectorReductionPhase:       // Vector reductions are over; get all the u and v infor
    
    DEBUGPR(2, rankTag << "vectorReductionPhase is complete");

    for (auto& node : localNodes)
      setUV(node);

    // debug
    if (!localNodes.empty()) {
      dbg_check_node_uv(myRank, verbosity, 4, localNodes.front(), scenProb, u, v, y,
                        localNodes.front().getUncondProb());
      if (localNodes.size() > 1) {
        dbg_check_node_uv(myRank, verbosity, 4, localNodes[1], scenProb, u, v, y,
                          localNodes[1].getUncondProb());
      }
    }    
    
    setCommPhase(scalarReductionPhase);
    break;

  case scalarReductionPhase:            // Scalar reduction is done and we're ready to project
    if (scalarReduceOutBuf[3] > 0.0) {
      if (myRank == 0) {
        std::cerr << "Terminating: numerical bailout detected on "
                  << static_cast<int>(scalarReduceOutBuf[3])
                  << " rank(s)" << std::endl;
      }
      return true;
    }
    cycleCount++;                       // Record one more completed cycle
    if (verbosity >= 1)
      global_toc("cycleCount increased to " \
                 + std::to_string(cycleCount) \
                 + " globalQueuedCount=" + std::to_string(globalQueuedCount) \
                 + " localQueuedCount=" + std::to_string(localQueuedCount) \
                 + " (immediately before a projectionStep call)"
                 );
    
    if (projectionStep())               // Do the projection step, computing new z and w
      return true;

    if (cycleCount > 1 && convergenceDetected()) {
      global_toc("Convergence Detected.");
      converged_ = true;
      return true;
    }

    // we will have done the projection step even if zero cycles are requested
    if (maxCyclesReached())
      return true;

    // Check external terminate flag (safe point: between comm phases)
    if (externalTerminateFlag_ && externalTerminateFlag_->load(std::memory_order_relaxed) != 0)
      return true;

    // Check stepping target (for runNCycles)
    if (cycleTarget_ > 0 && cycleCount >= cycleTarget_)
    {
      pausedAtCycleTarget_ = true;
      // Must drain inProjection to eligible before returning,
      // otherwise the next runNCycles call finds no eligible scenarios
      while(!inProjection.empty())
      {
        makeEligible(inProjection.front());
        inProjection.pop_front();
      }
      return true;
    }

    while(!inProjection.empty())        // Move formerly in-projection scenarios to eligible
    {
      makeEligible(inProjection.front());
      inProjection.pop_front();
    }
    setCommPhase(leadingSyncPhase);       // Indicate that it's time to start a fresh cycle

    break;

  default:

    throw std::runtime_error("Unknown finished communication phase " + std::to_string(commPhase));

  }

  // We have now handled all possible completed communication steps.
  // Initiate the next communication, based on the (new) value of commPhase.

  switch (commPhase)
  {
  case leadingSyncPhase:
 
    launchLeadingSync();
    break;      

  case vectorReductionPhase:

    DEBUGPR(2, rankTag << "vectorReductionPhase queuedForProjection.size()=" << queuedForProjection.size());
    
    while (!queuedForProjection.empty())
    {
      int s = queuedForProjection.front();
      scenario(s).fetchXinto(x[s]);        // Get the x value for this scenario
      scenario(s).fetchXExportInto(xExport[s]);  // and the candidate to export
      scenario(s).computeYinto(y[s]);      // Figure out y as well
      DEBUGPR(5, rankTag << "Computed for " << s << vectorInlineToString(" y[s]", y[s]));
      inProjection.push_back(s);           // This scenario is now in projection
      queuedForProjection.pop_front();     // and no longer queued
    }
    localQueuedCount = 0;

    for (auto& node : localNodes)
      launchNodeReduction(node);

    break;

  case scalarReductionPhase:

    for (auto &node : localNodes)
      launchScalarReduction(node);
    
    break;

  default:

    throw std::runtime_error("Unknown launch communication phase " + std::to_string(commPhase));

  }

  return false;

}


// Logical function that implements APH-specific rules for whether it is safe
// to dispatch a scenario

bool FWAPH::dispatchAllowed(void)
{
  // If dispatch is only allowed during leading sync and we're in the
  // vector or scalar reduction phases, prohibit dispatch

  if (leadingDispatchOnly && commPhase != leadingSyncPhase)
    return false;

  // If block-sync is on, then only allow just enough solves to start the
  // next projection

  if (blockSync && active.size() + queuedForProjection.size() >= localTrigger)
    return false;

  return true;
}

// Figure out the best scenario to send out next. This is the lexical max
// of (excess of staleness threshold, -phi)

std::pair<std::set<int>::iterator,int> FWAPH::bestTaskToDispatch(void)
{
    double smallestPhi    = INFINITY;
    int    worstStaleness = -1;

    auto toReturn = eligible.end();

    for (auto it = eligible.begin(); it != eligible.end(); ++it)
    {
        int s = *it;

        int staleness = std::max(0, cycleCount - lastDispatchCycle[s] - stalenessThreshhold);

        if ((staleness > worstStaleness) || 
            (staleness == worstStaleness && phi[s] < smallestPhi))
        {
            worstStaleness = staleness;
            smallestPhi = phi[s];
            toReturn = it;
        }
    }
    return {toReturn, FWsolveType};
}


// Enhance startTask so it keeps track of last dispatch time for FW solves
void FWAPH::startTask(int s, int taskType)
{
  applyPendingUpdates(s);
  if (taskType == FWsolveType)
    lastDispatchCycle[s] = cycleCount;
  groupController::startTask(s, taskType);
}


// Reduce the nodal part of x and y to get their averages.

void FWAPH::launchNodeReduction(ScenarioTreeNode& node) {
  int xLen = node.getDenseNonantLen();
  int naStart = node.getDenseNonantStart();

  auto &in  = node.inBuf();
  auto &out = node.outBuf();

  std::fill(in.begin(), in.end(), 0.0);

  for (int s : node.getNodeLocalScenarioIndexList()) {
    double prob = scenProb[s] / node.getUncondProb();
    for (int i = 0; i < xLen; ++i) {
      in[i]        += prob * x[s][naStart + i];
      in[i + xLen] += prob * y[s][naStart + i];
    }
  }

  int rc = MPI_Iallreduce(in.data(), out.data(),
                 2 * xLen, MPI_DOUBLE, MPI_SUM,
                 node.getComm(), nextFreeRequest());
  if (rc != MPI_SUCCESS)
    throw std::runtime_error("MPI_Iallreduce failed in launchNodeReduction");  
}


// v is ybar, which is after xbar in OutBuf
// (we have two thing going on: the buffer is shared between
// x and y; and the v vector has all nodes)
/*
  std::copy(nodeReduceOutBuf.begin() + xLen,
  nodeReduceOutBuf.end(),
  v.begin() + naStart);
*/
// Meanwhile, u is x - xbar
void FWAPH::computeUVforNode(ScenarioTreeNode& node) {
  int xLen = node.getDenseNonantLen();
  int naStart = node.getDenseNonantStart();
  auto &out = node.outBuf();

  // ---- Main updates: u = x - xbar, v = ybar ----
  for (int s : node.getNodeLocalScenarioIndexList()) {
    for (int i = 0; i < xLen; ++i) {
      u[s][naStart + i] = x[s][naStart + i] - out[i];   // out[0..xLen-1] is xbar
      v[s][naStart + i] = out[xLen + i];                // out[xLen..2*xLen-1] is ybar
    }
    DEBUGPR(4, rankTag << "Just reduced u for " << s << vectorInlineToString(" u[s]", u[s]));
    DEBUGPR(4, rankTag << "Just reduced v for " << s << vectorInlineToString(" v[s]", v[s]));
  }

}


void FWAPH::setUV(ScenarioTreeNode &node) {
  computeUVforNode(node);
}


// There are three things to reduce
void FWAPH::launchScalarReduction(ScenarioTreeNode& thisNode) {
  // We want GLOBAL phi and norms ONCE per comm cycle (not once per node).
  // Keep the signature to minimize refactoring, but only launch on the first node
  if (!localNodes.empty() && &thisNode != &localNodes.front()) {
    return;
  }

  double localPhiSum        = 0.0;
  double localSqNormU       = 0.0;  // probability-weighted squared norm (unweighted by omicron)
  double localSqNormV       = 0.0;
  double localSqNormU_Omega = 0.0;  // probability-weighted, omicron-weighted squared norm

  for (int s = 0; s < numSubdomains; ++s) {
    // phi uses the full vectors (all stages) once per scenario
    for (int i = 0; i < (int)z[s].size(); ++i) {
      localPhiSum += scenProb[s] * (z[s][i] - x[s][i]) * (W[s][i] - y[s][i]);
    }

    // norms also once per scenario (full vectors)
    localSqNormU += scenProb[s] * squaredNorm(u[s]);
    localSqNormV += scenProb[s] * squaredNorm(v[s]);

    // omicron-weighted u-norm: sum_i o_i * u_i^2, weighted by scenario prob.
    // Pre-freeze omicron is all-ones, so this equals localSqNormU until freeze.
    const auto& o = scenario(s).getOmicron();
    double wSq = 0.0;
    for (int i = 0; i < (int)u[s].size(); ++i) {
      wSq += o[i] * u[s][i] * u[s][i];
    }
    localSqNormU_Omega += scenProb[s] * wSq;
  }
  // --- DIAG: compare local phi vs local tau (rank 0) ---
  if (myRank == 0 && verbosity >= 3) {
    double localTau = localSqNormU_Omega + gammaInverse * localSqNormV;
    std::cerr << std::setprecision(17)
              << "[DIAG-LOCAL] phi=" << localPhiSum
              << " n2u=" << localSqNormU
              << " n2uOmega=" << localSqNormU_Omega
              << " n2v=" << localSqNormV
              << " tau=" << localTau
              << " phi-tau=" << (localPhiSum - localTau)
              << "\n";
  }
  scalarReduceInBuf[0] = localPhiSum;
  scalarReduceInBuf[1] = localSqNormU;
  scalarReduceInBuf[2] = localSqNormV;
  scalarReduceInBuf[3] = numericalBailoutDetected ? 1.0 : 0.0;
  scalarReduceInBuf[4] = localSqNormU_Omega;

  MPI_Iallreduce(scalarReduceInBuf.data(),
                 scalarReduceOutBuf.data(),
                 5,
                 MPI_DOUBLE,
                 MPI_SUM,
                 comm,
                 nextFreeRequest());
}

bool FWAPH::projectionStep()
{
  const double globalPhiSum        = scalarReduceOutBuf[0];
  const double globalSqNormU       = scalarReduceOutBuf[1];
  const double globalSqNormV       = scalarReduceOutBuf[2];
  const double globalSqNormU_Omega = scalarReduceOutBuf[4];

  // Note that the norms are probability weighted
  // (There is a completely unrelated tau in FW)
  // tau uses the omicron-weighted u-norm; normU (for termination) stays unweighted.
  const double tau = globalSqNormU_Omega + gammaInverse * globalSqNormV;

  normU = std::sqrt(globalSqNormU);
  normV = std::sqrt(globalSqNormV);
  DEBUGPR(2, rankTag << "normU=" << normU << " normV=" << normV
                     << " tau=" << tau
                     << " tauOmega(u part)=" << globalSqNormU_Omega);

  double theta = 0.0;
  if (tau > 0.0 && globalPhiSum > 0.0) {
    if (tau < tauTolerance) {
      if (myRank == 0) {
        std::cerr << "Terminating: tau=" << tau << " is below tauTolerance=" << tauTolerance << std::endl
                  << "To avoid this, try setting primal-term-thresh and dual-term-thresh higher, or aph-max-cycles lower"
                  << std::endl;
      }
      return true;
    }
    theta = (nu / tau) * globalPhiSum;
  }
  DEBUGPR(2, rankTag << "theta=" << theta);

  // for the W trace
  const double wallTime = global_seconds();

  // ---- Optional diagnostics (keep projectionStep readable) ----
  const bool doDiag = (myRank == 0 && verbosity >= 4);

  // pick a few scenarios to spot-check
  std::vector<int> sampleScens;
  if (doDiag && numSubdomains > 0) {
    sampleScens.push_back(0);
    if (numSubdomains > 2) sampleScens.push_back(numSubdomains / 2);
    if (numSubdomains > 1) sampleScens.push_back(numSubdomains - 1);
    std::sort(sampleScens.begin(), sampleScens.end());
    sampleScens.erase(std::unique(sampleScens.begin(), sampleScens.end()), sampleScens.end());
  }

  // Collective on the node communicators, so it must NOT be under doDiag (which
  // is rank-0 only); it gates itself on verbosity, which is uniform across ranks.
  diagCheckWInNPerp("before projection z/w update");

  // ---- Do the update ----
  for (int s = 0; s < numSubdomains; ++s) {
    const int n = static_cast<int>(z[s].size());
    const auto& o = scenario(s).getOmicron();

    // If these ever mismatch, something is very wrong; keep this cheap check always.
    if ((int)W[s].size() != n || (int)u[s].size() != n || (int)v[s].size() != n ||
        (int)o.size() != n) {
      std::cerr << "[FATAL] projectionStep size mismatch on rank " << myRank
                << " scen " << s
                << " z=" << z[s].size()
                << " W=" << W[s].size()
                << " u=" << u[s].size()
                << " v=" << v[s].size()
                << " o=" << o.size()
                << "\n";
      MPI_Abort(comm, 1);
    }

    // Optional: spot-check expected deltas on a couple indices BEFORE overwriting
    if (doDiag && std::binary_search(sampleScens.begin(), sampleScens.end(), s) && n > 0) {
      const int i0 = 0;
      const int i1 = (n > 1) ? (n - 1) : 0;

      const double W0_before = W[s][i0];
      const double z0_before = z[s][i0];
      const double W1_before = W[s][i1];
      const double z1_before = z[s][i1];

      // perform the update
      for (int i = 0; i < n; ++i) {
        z[s][i] += theta * gammaInverse * v[s][i];
        W[s][i] += theta * o[i] * u[s][i];
      }

      const double W0_exp = W0_before + theta * o[i0] * u[s][i0];
      const double z0_exp = z0_before + theta * gammaInverse * v[s][i0];
      const double W1_exp = W1_before + theta * o[i1] * u[s][i1];
      const double z1_exp = z1_before + theta * gammaInverse * v[s][i1];

      const double eW0 = std::abs(W[s][i0] - W0_exp);
      const double ez0 = std::abs(z[s][i0] - z0_exp);
      const double eW1 = std::abs(W[s][i1] - W1_exp);
      const double ez1 = std::abs(z[s][i1] - z1_exp);

      std::cerr << std::setprecision(17)
                << "[DIAG-UPDATE] scen=" << s
                << " theta=" << theta
                << " (i0=" << i0 << " eW=" << eW0 << " ez=" << ez0 << ")"
                << " (i1=" << i1 << " eW=" << eW1 << " ez=" << ez1 << ")"
                << "\n";
    } else {
      // normal path: just update
      for (int i = 0; i < n; ++i) {
        z[s][i] += theta * gammaInverse * v[s][i];
        W[s][i] += theta * o[i] * u[s][i];
      }
    }

    DEBUGPR(4, rankTag << "Just computed for " << s << vectorInlineToString(" z[s]", z[s]));
    DEBUGPR(4, rankTag << "Just computed for " << s << vectorInlineToString(" W[s]", W[s]));

    scenario(s).inplaceUpdateW(W[s]);
    scenario(s).inplaceUpdateZ(z[s]);

    if (wTrace_) {
      wTrace_->append(wallTime, scenario(s).getProblemName(), W[s]);
    }
  }

  // Collective; see the note at the "before projection" call above.
  diagCheckWInNPerp("right after projection z/w update");

  // ---- oscillation trace (rank 0, scenario 0) ----
  if (oscTraceFile_.is_open() && numSubdomains > 0) {
    oscTraceFile_ << cycleCount << "," << theta << "," << globalPhiSum
                  << "," << tau << "," << globalSqNormU_Omega
                  << "," << normU << "," << normV;
    const auto& oTrace = scenario(0).getOmicron();
    int nNonants = static_cast<int>(x[0].size());
    for (int i = 0; i < nNonants; ++i) {
      oscTraceFile_ << "," << x[0][i] << "," << z[0][i] << "," << u[0][i]
                    << "," << W[0][i] << "," << y[0][i] << "," << v[0][i]
                    << "," << oTrace[i];
    }
    oscTraceFile_ << "\n";
    oscTraceFile_.flush();
  }

  // ---- recompute phi[s] to use fresh data (used for dispatch) ----
  for (int s = 0; s < numSubdomains; ++s) {
    double ph = 0.0;
    for (int i = 0; i < (int)z[s].size(); ++i) {
      ph += scenProb[s] * (z[s][i] - x[s][i]) * (W[s][i] - y[s][i]);
    }
    phi[s] = ph;
  }

  return false;
}


// Common code to launch leading sync operations.  Put the local queued count in 
// a buffer and start a reduction to find put its global sum in globalQueuedCount

void FWAPH::launchLeadingSync(void)
{
  if (localQueuedCount < 0) {
    std::cerr << rankTag
              << "BUG: localQueuedCount is negative at launchLeadingSync: "
              << localQueuedCount << std::endl << std::flush;
    MPI_Abort(comm, 77);
  }

  DEBUGPR(1, rankTag << "FWAPH::launchLeadingSync with localQueuedCount="
                     << localQueuedCount);

  commQueuedCount = localQueuedCount;

  MPI_Iallreduce(&commQueuedCount,
                 &globalQueuedCount,
                 1,
                 MPI_INT,
                 MPI_SUM,
                 comm,
                 nextFreeRequest());
}


// Verify that the freshly-built W lies in the orthogonal complement of the
// nonanticipative subspace: for every tree node the probability-weighted sum of
// W over that node's scenarios must be zero.  APH requires this of its duals,
// and the --iter0-w-multiplier guess only satisfies it when rho is constant
// across the scenarios at each node (see the comment in startupSequence).
//
// This checks the property that actually matters rather than rho-constancy,
// which is merely sufficient for it, so it also catches a malformed rho file or
// any future path that sets W at startup.  The reduction runs on the node
// communicator, so unlike a purely local scan it sees scenarios on other ranks.
//
// `nodes` must be the same serial-number-sorted vector the caller uses for its
// other per-node collectives, so that every rank issues them in one order.
//
// On violation this aborts rather than throws: a node communicator need not
// span all ranks, so ranks outside the offending node would not see the bad
// residual and a throw could leave them waiting in a later collective.
// The local scenario lists cover only this rank, but a node's scenarios are
// spread over every rank in its communicator, so the weighted sum that decides
// whether W lies in the nonanticipative complement has to be reduced.  Returns
// the signed weighted sum per coordinate (which must vanish) in `residual`, and
// the weighted sum of magnitudes (which sets the scale a residual is judged
// against) in `scale`.
//
// Collective on the node communicator: every rank holding a scenario at this
// node must call it, and callers must visit nodes in serial-number order so the
// collectives line up.
void FWAPH::_reduceNodeWResidual(ScenarioTreeNode& node,
                                 std::vector<double>& residual,
                                 std::vector<double>& scale)
{
  const int xLen    = node.getDenseNonantLen();
  const int naStart = node.getDenseNonantStart();

  std::vector<double> inBuf(2 * xLen, 0.0);
  std::vector<double> outBuf(2 * xLen, 0.0);

  const auto& scenList = node.getNodeLocalScenarioIndexList();
  for (int s : scenList) {
    const double prob = scenProb[s] / node.getUncondProb();
    for (int i = 0; i < xLen; ++i) {
      const double wsi = W[s][naStart + i];
      inBuf[i]        += prob * wsi;
      inBuf[xLen + i] += prob * std::abs(wsi);
    }
  }

  MPI_Comm lcomm = node.getComm();
  if (!ensure_comm_ok("_reduceNodeWResidual pre-Allreduce", lcomm)) {
    throw std::runtime_error("bad node comm in FWAPH::_reduceNodeWResidual");
  }

  const int rc = MPI_Allreduce(inBuf.data(), outBuf.data(), 2 * xLen,
                               MPI_DOUBLE, MPI_SUM, lcomm);
  if (rc != MPI_SUCCESS) {
    throw std::runtime_error("bad result from Allreduce in FWAPH::_reduceNodeWResidual");
  }

  residual.assign(outBuf.begin(), outBuf.begin() + xLen);
  scale.assign(outBuf.begin() + xLen, outBuf.end());
}


// The local nodes in a globally consistent order, for issuing per-node blocking
// collectives.  Ranks hold different subsets of the nodes, but sorting by serial
// number means any two ranks sharing a pair of nodes agree on which comes first.
std::vector<ScenarioTreeNode*> FWAPH::_nodesInCollectiveOrder()
{
  std::vector<ScenarioTreeNode*> nodes;
  nodes.reserve(localNodes.size());
  for (auto& n : localNodes) nodes.push_back(&n);

  std::sort(nodes.begin(), nodes.end(),
            [](const ScenarioTreeNode* a, const ScenarioTreeNode* b) {
              return a->getSerialNumber() < b->getSerialNumber();
            });
  return nodes;
}


void FWAPH::_checkWOrthogonality(const std::vector<ScenarioTreeNode*>& nodes)
{
  // Residual is compared against the weighted mean magnitude of W at the same
  // coordinate, since an absolute tolerance is meaningless once W is large.
  const double relTol = 1e-9;

  std::vector<double> residualVec, scaleVec;

  for (ScenarioTreeNode* thisNode : nodes) {
    _reduceNodeWResidual(*thisNode, residualVec, scaleVec);

    const int xLen = static_cast<int>(residualVec.size());
    for (int i = 0; i < xLen; ++i) {
      const double residual = std::abs(residualVec[i]);
      const double scale    = scaleVec[i];
      if (residual > relTol * (1.0 + scale)) {
        global_toc("W ORTHOGONALITY VIOLATION at startup: node=" + thisNode->getName()
                   + " coord=" + std::to_string(i)
                   + " residual=" + std::to_string(residual)
                   + " scale=" + std::to_string(scale)
                   + " -- W = m*rho*(xbase-z0) left the nonanticipative"
                     " complement, which happens when rho differs across the"
                     " scenarios at this node; check the per-scenario"
                     " {scenarioName}_rho.csv files, or use --iter0-w-multiplier 0");
        MPI_Abort(comm, 1);
      }
    }
  }
}


//  Sequence to start things up
void FWAPH::startupSequence() {
  if (verbosity >= 1) global_toc("Begin Iter0 Lagrangian");
  runTaskOnAllSubdomains(LagrangianSolveType);

  for (int s = 0; s < numSubdomains; ++s) {
    scenario(s).fetchXinto(x[s]);
    scenario(s).fetchXExportInto(xExport[s]);
  }

  // ---- Compute z = xbar node-by-node ----
  // IMPORTANT: enforce a globally-consistent order of blocking collectives.
  std::vector<ScenarioTreeNode*> nodes = _nodesInCollectiveOrder();

  for (ScenarioTreeNode* thisNode : nodes) {
    const int xLen    = thisNode->getDenseNonantLen();
    const int naStart = thisNode->getDenseNonantStart();

    std::vector<double> inBuf(xLen, 0.0);
    std::vector<double> outBuf(xLen, 0.0);

    const auto& scenList = thisNode->getNodeLocalScenarioIndexList();
    for (int s : scenList) {
      const double prob = scenProb[s] / thisNode->getUncondProb();
      for (int i = 0; i < xLen; ++i) {
        inBuf[i] += prob * x[s][naStart + i];
      }
    }

    MPI_Comm lcomm = thisNode->getComm();
    if (!ensure_comm_ok("startupSequence xbar pre-Allreduce", lcomm)) {
      throw std::runtime_error("bad node comm pre-Allreduce in FWAPH::startupSequence");
    }

    const int rc = MPI_Allreduce(inBuf.data(),
                                 outBuf.data(),
                                 xLen,
                                 MPI_DOUBLE,
                                 MPI_SUM,
                                 lcomm);
    if (rc != MPI_SUCCESS) {
      char msg[MPI_MAX_ERROR_STRING]; int len = 0;
      MPI_Error_string(rc, msg, &len);
      std::cerr << "startupSequence MPI_Allreduce rc=" << rc << " : "
                << std::string(msg, len) << "\n";
      print_comm_info("Allreduce failed comm", lcomm);
      throw std::runtime_error("bad result from Allreduce in FWAPH::startupSequence");
    }

    // Push this node's slice of xbar into z for all scenarios at this node
    for (int s : scenList) {
      for (int i = 0; i < xLen; ++i) {
        z[s][naStart + i] = outBuf[i];
      }
      DEBUGPR(5, rankTag << "For " << s << " initial "
                         << vectorInlineToString("z[s]", z[s]));
    }
  }

  // ---- initialize W ----
  // W = 0 by default.  With --iter0-w-multiplier m > 0 we take the PH-style
  // guess W = m * rho * (xbase - z0) instead.
  //
  // APH requires W to lie in the orthogonal complement of the nonanticipative
  // subspace, i.e. the probability-weighted sum of W over the scenarios at each
  // node must vanish.  For this guess that holds only when rho is the same for
  // every scenario at the node, since only then can rho be pulled out of
  // sum_s p_s rho_s (x_s - z) to leave sum_s p_s (x_s - z) = 0.  rho is read
  // per scenario from {scenarioName}_rho.csv and aph-fw supports it varying by
  // scenario in general, so this is an assumption about the data, not an
  // invariant of the code -- _checkWOrthogonality below verifies it directly.
  for (int s = 0; s < numSubdomains; ++s) {
    if (iter0WMultiplier == 0.0) {
      std::fill(W[s].begin(), W[s].end(), 0.0);
    } else {
      const auto& rho = scenario(s).getRho();
      for (size_t i = 0; i < W[s].size(); ++i) {
        W[s][i] = iter0WMultiplier * rho[i] * (x[s][i] - z[s][i]);
      }
    }
    DEBUGPR(5, rankTag << "For " << s << " initial "
                       << vectorInlineToString("W[s]", W[s]));
  }

  // Only the m > 0 guess can leave the orthogonal complement; W = 0 is in it
  // trivially, so skip the collective in the default case.
  if (iter0WMultiplier != 0.0) {
    _checkWOrthogonality(nodes);
  }

  const double wallTime = global_seconds();

  for (int s = 0; s < numSubdomains; ++s) {
    scenario(s).inplaceUpdateZ(z[s]);
    scenario(s).inplaceUpdateW(W[s]);
    // Initialize y = 0 rather than y = W + rho(x - z); experiments
    // (experiments/y0_tests) found the two comparable, and y = 0 is simpler.
    // The startup y only matters under asynchronism: any scenario that
    // completes a solve gets y recomputed before its first reduction.
    // scenario(s).computeYinto(y[s]);
    std::fill(y[s].begin(), y[s].end(), 0.0);
    if (wTrace_) {
      wTrace_->append(wallTime, scenario(s).getProblemName(), W[s]);
    }
  }

  _computeCurrentLagrangianBound();

  global_toc("Iter0 Lagrangian bound is " + std::to_string(currentLagrangianBound));

  // ---- hand off to async mode ----
  //
  // The Lagrangian solve minimizes over K_i, not over conv K_i, so when the
  // original objective is nonlinear its minimizer xbase does not certify that
  // y = 0 is a subgradient of f_i at xbase: (xbase, 0) is not a valid pair.
  // For those problems we take one synchronous FW step on every scenario
  // first, using the usual subproblem formulation (W and prox terms, seeded
  // with xbase as the first vertex), which yields an (x0, y0) pair that is
  // exactly as valid as the ones produced at any later cycle.
  //
  // runTaskOnAllSubdomains blocks until every scenario is done, and each
  // completion goes through handleCompletedTask, so all scenarios land in
  // queuedForProjection.  The main event loop therefore sees a full queue and
  // projects immediately, computing y0 from x0 the same way it does for a
  // regular async subproblem (see the vectorReductionPhase launch branch).
  //
  // A linear objective needs none of this: x0 = xbase with y0 = 0 is already
  // a valid pair, so just start the FW solves and let the loop run.
  if (iter0FWStep) {
    if (verbosity >= 1) global_toc("Begin Iter0 FW step");
    runTaskOnAllSubdomains(FWsolveType);
    if (verbosity >= 1) global_toc("Iter0 FW step complete");
  } else {
    startTaskOnAllSubdomains(FWsolveType);
  }
}


// Shutdown sequence.  Do a Lagrangian solve to compute the final bound.

void FWAPH::terminationSequence(void)
{
  if (hubMode_) {
    // In hub mode, spokes normally compute bounds, so don't pay for the final
    // Lagrangian here.  The driver asks for it explicitly (via
    // computeFinalLagrangianBound) when --aphfw-final-lagrangian is set.
    if (verbosity >= 1) global_toc("Hub mode: skipping termination Lagrangian");
    return;
  }

  computeFinalLagrangianBound();
}


double FWAPH::computeFinalLagrangianBound()
{
  // Tell all scenarios to do a Lagrangian solve. This should start pushing in
  // solves as soon as any currently running FWSolves finish or abort
  // themselves.

  if (verbosity >= 1) global_toc("Begin Termination Lagrangian");

  // This will be done using the current W;
  //   however, the final scenario solutions are probably better.
  runTaskOnAllSubdomains(LagrangianSolveType);

  _computeCurrentLagrangianBound();

  global_toc("Termination Lagrangian bound is " + std::to_string(currentLagrangianBound));

  return currentLagrangianBound;
}


void FWAPH::startCommCycle(void) {
  setCommPhase(leadingSyncPhase);
  launchLeadingSync();
}

void FWAPH::printResults() {
  if (myRank == 0) {
    std::cout << "Current Lagrangian Bound: " << currentLagrangianBound
              << std::endl;
  }
  // for each scenario, output x and W
  if (csvOutputDirName != "") {
    for (int s = 0; s < numSubdomains; s++) {
      scenario(s).xToCsv(csvOutputDirName);
      scenario(s).WToCsv(csvOutputDirName);
    }
  }
}


void FWAPH::_computeCurrentLagrangianBound() {
  
  // declare the local part of the bound expectation
  std::array<double, 1> localExpectationSum{0.0}; 
  std::array<double, 1> globalExpectationBuf{0.0};
  
  // compute the local expectation
  for (int s = 0; s < numSubdomains; s++) {
    auto bndPair = scenario(s).getLagrangianResults();
    localExpectationSum[0] += scenProb[s] * bndPair.bnd;
  }

  // reduce (barrier) to get the expectation
  MPI_Allreduce(localExpectationSum.data(), globalExpectationBuf.data(), 1,
                MPI_DOUBLE, MPI_SUM, comm);

  // update the data
  currentLagrangianBound = globalExpectationBuf[0];
}


void FWAPH::saveWTrace()
{
  if (wTrace_) {
    wTrace_->save();
    wTrace_.reset();  // destroy it now, safely
  }
  if (solveTimesTrace_) {
    solveTimesTrace_->save();
    solveTimesTrace_.reset();
  }
}

// --- Stepping API ---

void FWAPH::onMainLoopTerminate()
{
  // In stepping mode, if we exited only because the cycle target was reached
  // preserve state so the caller can resume with another runNCycles call.
  // Every other exit (convergence, max cycles, the external flag, tau below
  // its tolerance, a numerical bailout) ends the run.
  if (pausedAtCycleTarget_) {
    clearCommActive();
  } else {
    signalTermination();
  }
}

// aph-max-cycles 0 means no limit when stepping with runNCycles (the Python
// hub); a standalone run stops after the first projection step.
bool FWAPH::maxCyclesReached() const
{
  if (aphMaxCycles <= 0 && cycleTarget_ > 0)
    return false;
  return cycleCount >= aphMaxCycles;
}

int FWAPH::runNCycles(int n)
{
  // If already at terminal state, don't re-enter the event loop
  int startCycle = cycleCount;
  cycleTarget_ = cycleCount + n;
  if (finished_ || converged_ || maxCyclesReached())
    return 0;
  pausedAtCycleTarget_ = false;

  if (!startupDone_) {
    startupSequence();
    startupDone_ = true;
  }

  // Reset terminateFlag so mainEventLoop can run again after a previous step
  terminateFlag = false;

  mainEventLoop();
  // Any exit but the pause at the cycle target, including the wall-clock limit
  // that returns from mainEventLoop directly, ends the run: the scenarios are
  // no longer eligible, so another event loop would wait forever.
  if (!pausedAtCycleTarget_)
    finished_ = true;

  // Don't call terminationSequence here — the caller decides when to finalize.
  // For hub mode, terminationSequence is a no-op anyway.
  // For stepping, we want to resume later without the expensive final Lagrangian.

  return cycleCount - startCycle;
}

// --- Data export ---

// A candidate solution: a point of K_s, which is what the xhat spokes and the
// denouement need.  For the algorithm's own iterate use getAllLocalProxX.
void FWAPH::getAllLocalNonants(double* buf, int len) const
{
  int offset = 0;
  for (int s = 0; s < numSubdomains; ++s) {
    int n = static_cast<int>(xExport[s].size());
    if (offset + n > len)
      throw std::runtime_error("getAllLocalNonants: buffer too small");
    std::copy(xExport[s].begin(), xExport[s].end(), buf + offset);
    offset += n;
  }
  if (offset != len)
    throw std::runtime_error("getAllLocalNonants: len mismatch (expected " +
                             std::to_string(offset) + ", got " + std::to_string(len) + ")");
}

// The proximal minimizer the algorithm iterates on.  This is what the dynamic
// rho extensions want: they measure dispersion of the scenario solutions, and
// exported vertices sit a fixed distance from a fractional consensus point
// however well converged the run is.
void FWAPH::getAllLocalProxX(double* buf, int len) const
{
  int offset = 0;
  for (int s = 0; s < numSubdomains; ++s) {
    int n = static_cast<int>(x[s].size());
    if (offset + n > len)
      throw std::runtime_error("getAllLocalProxX: buffer too small");
    std::copy(x[s].begin(), x[s].end(), buf + offset);
    offset += n;
  }
  if (offset != len)
    throw std::runtime_error("getAllLocalProxX: len mismatch (expected " +
                             std::to_string(offset) + ", got " + std::to_string(len) + ")");
}

// The probability-weighted mean of the scenario solutions, which is what PH
// calls xbar.  APH's z is the projection iterate and is a different object;
// getAllLocalXbar returns that one.  xbar = x - u, already computed per
// scenario, so this costs no new state.
void FWAPH::getAllLocalXmean(double* buf, int len) const
{
  int offset = 0;
  for (int s = 0; s < numSubdomains; ++s) {
    int n = static_cast<int>(x[s].size());
    if (offset + n > len)
      throw std::runtime_error("getAllLocalXmean: buffer too small");
    if (static_cast<int>(u[s].size()) != n)
      throw std::runtime_error("getAllLocalXmean: u[s] size differs from x[s]");
    for (int i = 0; i < n; ++i) buf[offset + i] = x[s][i] - u[s][i];
    offset += n;
  }
  if (offset != len)
    throw std::runtime_error("getAllLocalXmean: len mismatch (expected " +
                             std::to_string(offset) + ", got " + std::to_string(len) + ")");
}

void FWAPH::getAllLocalW(double* buf, int len) const
{
  int offset = 0;
  for (int s = 0; s < numSubdomains; ++s) {
    int n = static_cast<int>(W[s].size());
    if (offset + n > len)
      throw std::runtime_error("getAllLocalW: buffer too small");
    std::copy(W[s].begin(), W[s].end(), buf + offset);
    offset += n;
  }
  if (offset != len)
    throw std::runtime_error("getAllLocalW: len mismatch (expected " +
                             std::to_string(offset) + ", got " + std::to_string(len) + ")");
}

void FWAPH::getAllLocalXbar(double* buf, int len) const
{
  int offset = 0;
  for (int s = 0; s < numSubdomains; ++s) {
    int n = static_cast<int>(z[s].size());
    if (offset + n > len)
      throw std::runtime_error("getAllLocalXbar: buffer too small");
    std::copy(z[s].begin(), z[s].end(), buf + offset);
    offset += n;
  }
  if (offset != len)
    throw std::runtime_error("getAllLocalXbar: len mismatch (expected " +
                             std::to_string(offset) + ", got " + std::to_string(len) + ")");
}

void FWAPH::getAllLocalRho(double* buf, int len) const
{
  int offset = 0;
  for (int s = 0; s < numSubdomains; ++s) {
    const auto& rho = effectiveRho(s);
    int n = static_cast<int>(rho.size());
    if (offset + n > len)
      throw std::runtime_error("getAllLocalRho: buffer too small");
    std::copy(rho.begin(), rho.end(), buf + offset);
    offset += n;
  }
  if (offset != len)
    throw std::runtime_error("getAllLocalRho: len mismatch (expected " +
                             std::to_string(offset) + ", got " + std::to_string(len) + ")");
}

const std::vector<double>& FWAPH::effectiveRho(int s) const
{
  if (s < (int)pendingRho_.size() && pendingRho_[s])
    return *pendingRho_[s];
  return localScenarios()[s].getRho();
}

std::pair<const std::vector<double>&, const std::vector<double>&>
FWAPH::effectiveNonantBounds(int s) const
{
  if (s < (int)pendingBounds_.size() && pendingBounds_[s])
    return {pendingBounds_[s]->first, pendingBounds_[s]->second};
  return {localScenarios()[s].getNonantLowerBounds(),
          localScenarios()[s].getNonantUpperBounds()};
}

void FWAPH::applyPendingUpdates(int s)
{
  if (s < (int)pendingRho_.size() && pendingRho_[s]) {
    localScenarios()[s].setRho(*pendingRho_[s]);
    pendingRho_[s].reset();
  }
  if (s < (int)pendingBounds_.size() && pendingBounds_[s]) {
    localScenarios()[s].setNonantBounds(pendingBounds_[s]->first,
                                        pendingBounds_[s]->second);
    pendingBounds_[s].reset();
  }
}

void FWAPH::setAllLocalRho(const double* buf, int len)
{
  int offset = 0;
  for (int s = 0; s < numSubdomains; ++s) {
    int n = localScenarios()[s].getTotalNonantCount();
    if (offset + n > len)
      throw std::runtime_error("setAllLocalRho: buffer too small");
    std::vector<double> newRho(buf + offset, buf + offset + n);
    // The hub pushes rho every cycle, mostly unchanged; an unchanged one must
    // not touch the solver objects (it would rebuild their objectives).
    // What the solvers already have: drop any held update, write nothing.
    if (newRho == localScenarios()[s].getRho()) {
      if (s < (int)pendingRho_.size())
        pendingRho_[s].reset();
      offset += n;
      continue;
    }
    if (newRho == effectiveRho(s)) {  // the update already held
      offset += n;
      continue;
    }
    if (isInSet(s, active)) {
      pendingRho_.resize(numSubdomains);
      pendingRho_[s] = std::move(newRho);
    } else {
      localScenarios()[s].setRho(newRho);
      // a held older update must not be applied over this one
      if (s < (int)pendingRho_.size())
        pendingRho_[s].reset();
    }
    offset += n;
  }
  if (offset != len)
    throw std::runtime_error("setAllLocalRho: len mismatch (expected " +
                             std::to_string(offset) + ", got " + std::to_string(len) + ")");
}

void FWAPH::getAllLocalOmicron(double* buf, int len) const
{
  int offset = 0;
  for (int s = 0; s < numSubdomains; ++s) {
    const auto& o = localScenarios()[s].getOmicron();
    int n = static_cast<int>(o.size());
    if (offset + n > len)
      throw std::runtime_error("getAllLocalOmicron: buffer too small");
    std::copy(o.begin(), o.end(), buf + offset);
    offset += n;
  }
  if (offset != len)
    throw std::runtime_error("getAllLocalOmicron: len mismatch (expected " +
                             std::to_string(offset) + ", got " + std::to_string(len) + ")");
}

bool FWAPH::omicronIsFrozen() const
{
  for (int s = 0; s < numSubdomains; ++s) {
    if (!localScenarios()[s].omicronIsFrozen()) return false;
  }
  return true;
}

void FWAPH::freezeOmicron(double multiplier)
{
  // One-shot: throw if any local scenario's omicron is already frozen.
  for (int s = 0; s < numSubdomains; ++s) {
    if (localScenarios()[s].omicronIsFrozen()) {
      throw std::runtime_error("FWAPH::freezeOmicron called but scenario " +
                               localScenarios()[s].getProblemName() +
                               " already has frozen omicron.");
    }
  }

  // Compute and install omicron = multiplier * rho for each local scenario.
  double localMin = std::numeric_limits<double>::infinity();
  double localMax = -std::numeric_limits<double>::infinity();
  double localSum = 0.0;
  int    localCnt = 0;
  for (int s = 0; s < numSubdomains; ++s) {
    const auto& rhoS = effectiveRho(s);
    std::vector<double> o(rhoS.size());
    for (size_t i = 0; i < rhoS.size(); ++i) {
      o[i] = multiplier * rhoS[i];
      if (o[i] < localMin) localMin = o[i];
      if (o[i] > localMax) localMax = o[i];
      localSum += o[i];
    }
    localCnt += (int)rhoS.size();
    localScenarios()[s].setOmicron(o);  // bounds-check + sets scenario's frozen flag
  }

  // Optional S(N)-constancy check across local scenarios sharing each tree node.
  // (rho already satisfies this by construction when pushed via the mpi-sppy bridge;
  //  this catches accidental decoupling.)
  bool checkInvariant = (numSubdomains > 0) &&
                        localScenarios()[0].getCheckOmicronInvariant();
  if (checkInvariant) {
    for (auto& node : localNodes) {
      const auto& scenList = node.getNodeLocalScenarioIndexList();
      if (scenList.size() < 2) continue;
      int start = node.getDenseNonantStart();
      int len   = node.getDenseNonantLen();
      int ref = scenList[0];
      const auto& oRef = localScenarios()[ref].getOmicron();
      for (size_t k = 1; k < scenList.size(); ++k) {
        int si = scenList[k];
        const auto& oSi = localScenarios()[si].getOmicron();
        for (int j = 0; j < len; ++j) {
          double a = oRef[start + j];
          double b = oSi[start + j];
          double tol = 1e-12 * (1.0 + std::abs(a) + std::abs(b));
          if (std::abs(a - b) > tol) {
            global_toc("OMICRON INVARIANT VIOLATION: node=" + node.getName() +
                       " coord=" + std::to_string(j) +
                       " scenRef=" + std::to_string(ref) +
                       " scenOther=" + std::to_string(si) +
                       " a=" + std::to_string(a) + " b=" + std::to_string(b));
            MPI_Abort(comm, 1);
          }
        }
      }
    }
  }

  double localMean = (localCnt > 0) ? (localSum / localCnt) : 0.0;
  global_toc("omicron frozen: multiplier=" + std::to_string(multiplier) +
             " cycle=" + std::to_string(cycleCount) +
             " localMin=" + std::to_string(localMin) +
             " localMax=" + std::to_string(localMax) +
             " localMean=" + std::to_string(localMean));
}

void FWAPH::getAllLocalNonantBounds(double* lbBuf, double* ubBuf, int len) const
{
  int offset = 0;
  for (int s = 0; s < numSubdomains; ++s) {
    const auto [lb, ub] = effectiveNonantBounds(s);
    int n = static_cast<int>(lb.size());
    if (offset + n > len)
      throw std::runtime_error("getAllLocalNonantBounds: buffer too small");
    for (int i = 0; i < n; ++i) {
      lbBuf[offset + i] = lb[i];
      ubBuf[offset + i] = ub[i];
    }
    offset += n;
  }
  if (offset != len)
    throw std::runtime_error("getAllLocalNonantBounds: len mismatch (expected " +
                             std::to_string(offset) + ", got " + std::to_string(len) + ")");
}

void FWAPH::setAllLocalNonantBounds(const double* lbBuf, const double* ubBuf, int len)
{
  int offset = 0;
  for (int s = 0; s < numSubdomains; ++s) {
    int n = localScenarios()[s].getTotalNonantCount();
    if (offset + n > len)
      throw std::runtime_error("setAllLocalNonantBounds: buffer too small");
    std::vector<double> lb(lbBuf + offset, lbBuf + offset + n);
    std::vector<double> ub(ubBuf + offset, ubBuf + offset + n);
    // As for rho: the hub pushes every bound every cycle, and rewriting an
    // unchanged bound can discard a backend's MIP start.
    // What the solvers already have: drop any held update, write nothing.
    if (lb == localScenarios()[s].getNonantLowerBounds() &&
        ub == localScenarios()[s].getNonantUpperBounds()) {
      if (s < (int)pendingBounds_.size())
        pendingBounds_[s].reset();
      offset += n;
      continue;
    }
    const auto [heldLb, heldUb] = effectiveNonantBounds(s);
    if (lb == heldLb && ub == heldUb) {  // the update already held
      offset += n;
      continue;
    }
    if (isInSet(s, active)) {
      pendingBounds_.resize(numSubdomains);
      pendingBounds_[s] = std::make_pair(std::move(lb), std::move(ub));
    } else {
      localScenarios()[s].setNonantBounds(lb, ub);
      // a held older update must not be applied over this one
      if (s < (int)pendingBounds_.size())
        pendingBounds_[s].reset();
    }
    offset += n;
  }
  if (offset != len)
    throw std::runtime_error("setAllLocalNonantBounds: len mismatch (expected " +
                             std::to_string(offset) + ", got " + std::to_string(len) + ")");
}

std::vector<std::string> FWAPH::getLocalScenarioNames() const
{
  std::vector<std::string> names;
  names.reserve(numSubdomains);
  for (int s = 0; s < numSubdomains; ++s)
    names.push_back(localScenarios()[s].getProblemName());
  return names;
}

int FWAPH::getLocalNonantLength() const
{
  int total = 0;
  for (int s = 0; s < numSubdomains; ++s)
    total += static_cast<int>(x[s].size());
  return total;
}


// A function for debugging: report, per node, how far W is from the
// nonanticipative complement.
//
// This is collective on each node's communicator, so EVERY rank has to call it
// -- do not guard the call with myRank == 0.  The only gate is verbosity, which
// holds the same value on every rank.  Reducing rather than scanning rank-local
// scenario lists is what makes the number meaningful: a node's scenarios are
// spread across ranks, and with one scenario per rank a local mean is just that
// scenario's own W, so a local version reports a violation for any nonzero W.
//
// Rank 0 does the printing, for its own nodes; the other ranks contribute to
// the reductions silently.
void FWAPH::diagCheckWInNPerp(const std::string& tag)
{
  if (verbosity < 4) return;

  // Same relative test as _checkWOrthogonality, one order of magnitude looser.
  const double relTol = 1e-8;

  if (myRank == 0) {
    std::cerr << "[DIAG-NPERP] --- " << tag << " ---\n";
  }

  std::vector<double> residualVec, scaleVec;

  for (ScenarioTreeNode* node : _nodesInCollectiveOrder()) {
    _reduceNodeWResidual(*node, residualVec, scaleVec);

    if (myRank != 0) continue;   // reduction done; only rank 0 reports

    const int naStart = node->getDenseNonantStart();
    const int xLen    = static_cast<int>(residualVec.size());

    double worstAbsMean = 0.0;
    double worstScale   = 0.0;
    int    worstJ       = -1;

    for (int j = 0; j < xLen; ++j) {
      const double absMean = std::abs(residualVec[j]);
      if (absMean > worstAbsMean) {
        worstAbsMean = absMean;
        worstScale   = scaleVec[j];
        worstJ       = j;
      }
    }

    if (worstAbsMean > relTol * (1.0 + worstScale)) {
      std::cerr << std::setprecision(17)
                << "[DIAG-NPERP] ^^^ Node=" << node->getName()
                << " worstAbsMeanW=" << worstAbsMean
                << " scaleThere=" << worstScale
                << " worstLocalIndex=" << worstJ
                << " globalIndex=" << (naStart + (worstJ < 0 ? 0 : worstJ))
                << "  (EXPECTED " << (relTol * (1.0 + worstScale)) << " or less)\n";
    }
    else {
      std::cerr << "[DIAG-NPERP] Node=" << node->getName()
                << " OK (worstAbsMeanW=" << worstAbsMean << ")\n";
    }
  }

  if (myRank == 0) {
    std::cerr << "[DIAG-NPERP] ---------------------------\n";
  }
}
