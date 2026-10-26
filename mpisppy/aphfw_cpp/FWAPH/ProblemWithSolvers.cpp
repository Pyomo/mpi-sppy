///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "ProblemWithSolvers.hpp"

ProblemWithSolvers::~ProblemWithSolvers() = default;

void ProblemWithSolvers::registerOptions(Config& config) { // static
  config.addOption<std::string>("problem-name", "", "Name of the problem (will be used as the scenario directory name)");
  config.addOption<std::string>("iter0-json", "", "Iter0 parameters JSON file");

  // rho inputs:
  config.addOption<std::string>("rho-csv", "", "Override rho CSV path (default: derive {scenarioName}_rho.csv; ignored if --rho-all is provided)");
  config.addOption<double>("rho-all", -1.0, "If >= 0, set rho for ALL variables to this value and skip reading rho-csv");

  // omicron (dual-side metric weights). Default: freeze omicron = multiplier * rho after iteration 0.
  config.addOption<double>("omicron-rho-multiplier", 1.0, "At post-iter-0 freeze, set omicron = k * rho with this k (default 1.0)");
  config.addOption<double>("omicron-all", -1.0, "If >= 0, freeze omicron to this scalar at setup (overrides --omicron-csv and --omicron-rho-multiplier)");
  config.addOption<std::string>("omicron-csv", "", "If non-empty, freeze omicron from this CSV at setup (overrides --omicron-rho-multiplier)");
  config.addOption<double>("omicron-min", 1e-6, "Lower bound for omicron entries; throws if violated at freeze time");
  config.addOption<double>("omicron-max", 1e6, "Upper bound for omicron entries; throws if violated at freeze time");
  config.addOption<bool>("check-omicron-invariant", true, "Run S(N)-constancy check on omicron at freeze moment");

  config.addOption<std::string>("nl-json", "", "NL parameters JSON file");
  config.addOption<std::string>("grad-json", "", "grad parameters JSON file");
  config.addOption<std::string>("solver-log-file", "", "solver logfile (drops gurobi stdout as side-effect)");
  config.addOption<int>("FW-tmax", 1, "Max grad solves in FW");
  config.addOption<double>("FW-tau", 0, "FW convegence criterion");
  config.addOption<bool>("FW-obj-cutoff", false, "Add an objective cutoff row to FW MIP solves, as mpi-sppy's FWPH does (forces an improving direction)");
  config.addOption<double>("FW-obj-cutoff-tol", 1e-4, "Slack of the FW objective cutoff, relative to its value with this as a floor (mpi-sppy FWPH's fwph_conv_thresh, default 1e-4)");
  config.addOption<std::string>("FW-export-policy", "best-column",
                                "Candidate solution to export: best-column, last-vertex or prox "
                                "(the algorithm always gets the proximal minimizer)");
  config.addOption<double>("obj-scale", 1.0, "Multiply all objective coefficients by this value (default 1.0, no scaling)");
  config.addOption<bool>("abort-inflight-solves", true, "Interrupt in-flight subproblem solves when termination is detected");
}


void ProblemWithSolvers::setup(Config& cfg,
                               const std::string& pName,
                               std::vector<double>& W,
                               std::vector<double>& z) {
  problemName = pName;
  W_ = &W;
  z_ = &z;
  _getFromCfg(cfg);

  _createInterfaceObjects(cfg);

  // invariant: the derived solver interface should have built nonant maps by now
  if (Lagrangian->nonantNamesVector().empty()) {
    throw std::logic_error("Programming error: Lagrangian nonant maps not initialized after construction");
  }
  
  // Optional objective scaling (must happen before rho/addXSquared)
  double objScale = 1.0;
  getConfigOptionOrThrow<double>("obj-scale", objScale, cfg);
  if (objScale != 1.0) {
    Lagrangian->scaleObjective(objScale);
    nl->scaleObjective(objScale);
    gradient->scaleObjective(objScale);
    Lagrangian->fullObjectiveRebuild();
    gradient->fullObjectiveRebuild();
    // NL rebuild happens in addXSquared below
  }

  _initRho();
  updateRho(rhoAsRead);

  // Initialize omicron to all-ones. Cycle 1's projection therefore runs with
  // identity metric (baseline EWW). The default post-iter-0 freeze path
  // overwrites this via FWAPH::freezeOmicron; the setup-time override paths
  // (--omicron-all / --omicron-csv) freeze it here.
  omicron.assign(totalNonantCount, 1.0);
  if (omicronAll_.has_value()) {
    std::vector<double> o(totalNonantCount, *omicronAll_);
    setOmicron(o);
  } else if (!omicronCsv_.empty()) {
    std::vector<double> o;
    _readOmicron(o);
    setOmicron(o);
  }
  // else: leave at all-ones; FWAPH::freezeOmicron will install values after iter 0.

  // Capture initial variable bounds from the MPS file so they can be
  // exported to Python and later updated by the reduced-costs bridge.
  Lagrangian->captureInitialBounds();
  gradient->captureInitialBounds();

  nl->addXSquared();

#ifdef USE_GUROBI
  FW = std::make_unique<FrankWolfe<GurobiNL, GurobiGrad>>(*nl, *gradient, cfg);
#endif
#ifdef USE_CPLEX
  FW = std::make_unique<FrankWolfe<CPLEXNL, CPLEXGrad>>(*nl, *gradient, cfg);
#endif
#ifdef USE_XPRESS
  FW = std::make_unique<FrankWolfe<XpressNL, XpressGrad>>(*nl, *gradient, cfg);
#endif

  setupWasCalled = true;
}

void ProblemWithSolvers::_getFromCfg(Config& cfg) {
  // Optional override:
  double rhoAllRaw = -1.0;
  getConfigOptionOrThrow<double>("rho-all", rhoAllRaw, cfg);
  if (rhoAllRaw >= 0.0) {
    rhoAll = rhoAllRaw;
  } else {
    rhoAll.reset();
  }

  // rho-csv: if explicitly provided, use it; otherwise derive from scenario name.
  // (--rho-all takes priority over both and skips CSV reading entirely.)
  if (!rhoAll.has_value()) {
    std::string explicitRhoCsv;
    getConfigOptionOrThrow<std::string>("rho-csv", explicitRhoCsv, cfg);
    if (!explicitRhoCsv.empty()) {
      rhoCsv = explicitRhoCsv;
    } else {
      // Convention: {scenarioName}_rho.csv alongside the .mps and _nonants.json
      rhoCsv = problemName + "_rho.csv";
    }
  } else {
    rhoCsv = ""; // unused
  }

  getConfigOptionOrThrow<bool>("abort-inflight-solves", abortInFlightEnabled_, cfg);

  iter0Json = cfg.getRequiredString("iter0-json");
  nlJson = cfg.getRequiredString("nl-json");
  gradJson = cfg.getRequiredString("grad-json");

  // The logfile can be empty (if you don't want logging)
  getConfigOptionOrThrow<std::string>("solver-log-file", logFileName, cfg);
  getConfigOptionOrThrow<int>("FW-tmax", tMax, cfg);
  getConfigOptionOrThrow<double>("FW-tau", tau, cfg);

  // omicron options
  getConfigOptionOrThrow<double>("omicron-rho-multiplier", omicronRhoMultiplier_, cfg);
  double omicronAllRaw = -1.0;
  getConfigOptionOrThrow<double>("omicron-all", omicronAllRaw, cfg);
  if (omicronAllRaw >= 0.0) {
    omicronAll_ = omicronAllRaw;
  } else {
    omicronAll_.reset();
  }
  getConfigOptionOrThrow<std::string>("omicron-csv", omicronCsv_, cfg);
  getConfigOptionOrThrow<double>("omicron-min", omicronMin_, cfg);
  getConfigOptionOrThrow<double>("omicron-max", omicronMax_, cfg);
  getConfigOptionOrThrow<bool>("check-omicron-invariant", checkOmicronInvariant_, cfg);
}

void ProblemWithSolvers::_readRho() {
  DEBUGPR(1, "Reading " + rhoCsv);

  auto rhoMap = readCsvToMapWithHeaders(rhoCsv, "varname", "rho");

  const auto nNonants = Lagrangian->nonantNamesVector().size();
  if (rhoMap.size() != nNonants) {
    DEBUGPR(0, "WARNING: rhoMap.size()=" + std::to_string(rhoMap.size()) +
                   " != nonants=" + std::to_string(nNonants) +
                   " (rho-csv may be missing/excess entries)");
  }

  rhoAsRead = vectorFromMapWithOrder(rhoMap, Lagrangian->nonantNamesVector());
}


void ProblemWithSolvers::_initRho() {
  if (rhoAll.has_value()) {
    // totalNonantCount is known after _createInterfaceObjects()
    rhoAsRead.assign(totalNonantCount, *rhoAll);
  } else {
    _readRho();
  }
}


 void ProblemWithSolvers::updateRho(const std::vector<double>& newRho) {
   //printVectorInline("newRho", newRho);
   rho = newRho;
   Lagrangian->set_rho(newRho);
   nl->set_rho(newRho);
   gradient->set_rho(newRho);
}

void ProblemWithSolvers::setOmicron(const std::vector<double>& newO) {
  if (omicronFrozen_) {
    throw std::runtime_error("ProblemWithSolvers::setOmicron called after omicron was already frozen "
                             "(scenario " + problemName + "). Omicron is one-shot.");
  }
  if ((int)newO.size() != totalNonantCount) {
    throw std::runtime_error("ProblemWithSolvers::setOmicron size mismatch for scenario " +
                             problemName + ": got " + std::to_string(newO.size()) +
                             ", expected " + std::to_string(totalNonantCount));
  }
  for (int i = 0; i < (int)newO.size(); ++i) {
    if (!(newO[i] >= omicronMin_) || !(newO[i] <= omicronMax_)) {
      throw std::runtime_error("omicron out-of-bounds for scenario " +
                               problemName + " index " + std::to_string(i) +
                               ": value=" + std::to_string(newO[i]) +
                               " not in [" + std::to_string(omicronMin_) + ", " +
                               std::to_string(omicronMax_) + "]");
    }
  }
  omicron = newO;
  omicronFrozen_ = true;
}

void ProblemWithSolvers::_readOmicron(std::vector<double>& out) {
  DEBUGPR(1, "Reading " + omicronCsv_);
  auto omap = readCsvToMapWithHeaders(omicronCsv_, "varname", "omicron");
  const auto nNonants = Lagrangian->nonantNamesVector().size();
  if (omap.size() != nNonants) {
    DEBUGPR(0, "WARNING: omicronMap.size()=" + std::to_string(omap.size()) +
                   " != nonants=" + std::to_string(nNonants) +
                   " (omicron-csv may be missing/excess entries)");
  }
  out = vectorFromMapWithOrder(omap, Lagrangian->nonantNamesVector());
}

void ProblemWithSolvers::updateNonantBounds(const std::vector<double>& lb,
                                             const std::vector<double>& ub) {
  Lagrangian->setNonantBounds(lb, ub);
  // NL deliberately skipped — see design/reduced-costs-bridge.md
  gradient->setNonantBounds(lb, ub);
}

void ProblemWithSolvers::_createInterfaceObjects(Config& cfg) {
#ifdef USE_CPLEX
  DEBUGPR(1, "Using CPLEX.");

  iter0Params    = std::make_unique<CPLEXParams>(iter0Json);
  nlParams       = std::make_unique<CPLEXParams>(nlJson);
  gradientParams = std::make_unique<CPLEXParams>(gradJson);

  Lagrangian = std::make_unique<CPLEXInterface>(problemName, logFileName);
  nl         = std::make_unique<CPLEXNL>(problemName, logFileName);
  gradient   = std::make_unique<CPLEXGrad>(problemName, logFileName);
#endif

#ifdef USE_GUROBI
  DEBUGPR(1, "Using Gurobi.");

  iter0Params    = std::make_unique<GurobiParams>(iter0Json);
  nlParams       = std::make_unique<GurobiParams>(nlJson);
  gradientParams = std::make_unique<GurobiParams>(gradJson);

  Lagrangian = std::make_unique<GurobiInterface>(problemName, logFileName);
  nl         = std::make_unique<GurobiNL>(problemName, logFileName);
  gradient   = std::make_unique<GurobiGrad>(problemName, logFileName);
#endif

#ifdef USE_XPRESS
  DEBUGPR(1, "Using Xpress.");

  iter0Params    = std::make_unique<XpressParams>(iter0Json);
  nlParams       = std::make_unique<XpressParams>(nlJson);
  gradientParams = std::make_unique<XpressParams>(gradJson);

  Lagrangian = std::make_unique<XpressInterface>(problemName, logFileName);
  nl         = std::make_unique<XpressNL>(problemName, logFileName);
  gradient   = std::make_unique<XpressGrad>(problemName, logFileName);
#endif

  if (!iter0Params || !nlParams || !gradientParams) {
    throw std::runtime_error("Solver parameter objects were not created.");
  }
  if (!Lagrangian || !nl || !gradient) {
    throw std::runtime_error("Solver interface objects were not created.");
  }

  iter0Params->pushSolverParams(*Lagrangian);
  nlParams->pushSolverParams(*nl);
  gradientParams->pushSolverParams(*gradient);

  totalNonantCount = Lagrangian->getTotalNonantCount();
  globalNodeCount  = Lagrangian->getGlobalNodeCount();

  W_->assign(totalNonantCount, 0.0);
  z_->resize(totalNonantCount);

  Lagrangian->set_proxOn(false);
  Lagrangian->set_WOn(true);

  nl->attach_z(*z_);
  gradient->attach_z(*z_);

  Lagrangian->attach_W(*W_);
  nl->attach_W(*W_);
  gradient->attach_W(*W_);

  // prox and W both need to be on for nl
  nl->set_proxOn(true);
  nl->set_WOn(true);
}


objAndBnd ProblemWithSolvers::doLagrangianSolve(std::vector<double>& x) {
  assert(setupWasCalled && "setup must be called for ProblemWithSolvers");

  if (Lagrangian->isProxOn()) {
    DEBUGPR(0, "WARNING: Lagrangian solve called with ProxOn");
  }

  // Keep this off by default; enable only when you really need it.
#ifdef PWS_LAGR_CALL_DIAG
  {
    auto *L = Lagrangian.get();
    if (L) {
      std::cerr << "[LAGR-CALL] scenario=" << L->getScenarioName()
                << " proxOn=" << (L->isProxOn() ? "Y" : "N")
                << " WOn=" << (L->isWOn() ? "Y" : "N");

      if (L->isWOn()) {
        if (auto wp = L->get_WPtr(); wp && !wp->empty()) {
          std::cerr << " example W[0]=" << (*wp)[0];
        }
      }
      if (auto zp = L->get_zPtr(); zp && !zp->empty()) {
        std::cerr << " z[0]=" << (*zp)[0];
      }
      std::cerr << std::endl;
    } else {
      std::cerr << "[LAGR-CALL] ERROR: Lagrangian pointer is NULL\n";
    }
  }
#endif

  Lagrangian->fullObjectiveRebuild();

  if (Lagrangian->hasNumericalBailout()) {
    std::cerr << "Numerical bailout detected in fullObjectiveRebuild for '"
              << problemName << "'\n";
    return {0.0, 0.0};
  }

  auto [obj, bnd] = Lagrangian->solve();

  if (Lagrangian->wasInterrupted()) {
    // Shutdown in progress: there may be no solution to extract, and the
    // caller discards the result anyway.
    return {0.0, 0.0};
  }

  Lagrangian->extractNonantSolution(x);
  fullX = Lagrangian->copyEntireSolution();

  if (Lagrangian->hasNumericalBailout()) {
    std::cerr << "Numerical bailout detected in Lagrangian solve for '"
              << problemName << "'\n";
  }

  return {obj, bnd};
}


void ProblemWithSolvers::attachAbortFlag(const std::atomic<bool>* flag) {
  if (!setupWasCalled) {
    throw std::logic_error("attachAbortFlag called before setup()");
  }
  if (!abortInFlightEnabled_) {
    return;
  }
  abortFlagPtr_ = flag;
  Lagrangian->attachAbortFlag(flag);
  nl->attachAbortFlag(flag);
  gradient->attachAbortFlag(flag);
  FW->setAbortFlag(flag);
}

void ProblemWithSolvers::interruptInFlightSolves() {
  if (!abortInFlightEnabled_) {
    return;
  }
  if (Lagrangian) Lagrangian->interruptSolve();
  if (nl) nl->interruptSolve();
  if (gradient) gradient->interruptSolve();
}

void ProblemWithSolvers::doFWsolve(std::vector<double>& x,
                                  std::vector<double>& xExport) {
  if (abortFlagPtr_ && abortFlagPtr_->load(std::memory_order_relaxed)) {
    return;  // shutdown in progress; both keep their old values and are discarded
  }
  if (FW->get_R().size() == 0) {  // first time caller
    if (fullX.size() == 0) {
      throw std::runtime_error(
                               "FWsolve called for the first time, but fullX is empty.\n"
                               "(You may need to do a Lagrangian solve first.)"
                               );
    }
    auto r = fullX;  // assume that something (e.g., Lagrangian) got this for us
    FW->go(r, tMax, tau);
  } else {
    FW->goAgain(tMax, tau);
  }
  x = FW->getFinalNonantSolution();        // the proximal minimizer
  xExport = FW->getFinalVertexSolution();  // a point of K_s, for the spokes
}


// The next function is for testing (if you call it, the Lagrangian will not
//  be the Lagrangian anymore)
void ProblemWithSolvers::enableLagrangianProxAt(std::vector<double>& z) {
    if (!Lagrangian) {
        throw std::logic_error("enableLagrangianProxAt called before setup()");
    }

    // Set the prox center and enable the quadratic prox term
    Lagrangian->attach_z(z);
    Lagrangian->set_proxOn(true);
    Lagrangian->set_WOn(true);  // you might want W=0
    Lagrangian->addXSquared();  // this puts in the quad value    
}

void ProblemWithSolvers::writeLagrangianModel(const std::string& filename) const {
    if (!Lagrangian) {
        throw std::logic_error("writeLagrangianModel called before setup()");
    }
    Lagrangian->exportModel(filename);
}
