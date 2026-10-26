#include "scenario.hpp"
#include "utils/timer.hpp"

void Scenario::setup(Config& cfg, const std::string& scenName) {
  problemSolver.setup(cfg, scenName, W, z);
  x.resize(problemSolver.getTotalNonantCount());
  xExport.resize(problemSolver.getTotalNonantCount());
  problemSolver.attachAbortFlag(&abortFlag);
  setupWasCalled = true;
}

void Scenario::signalAbort(void) {
  subdomain::signalAbort();
  problemSolver.interruptInFlightSolves();
}

void Scenario::performTask(void)
{
  switch (taskType)
  {
    case LagrangianSolveType:
      lagrangianResults = LagrangianSolve();
      break;

    case FWsolveType:
      FWsolve();
      break;

    default:
      throw std::runtime_error("Bad task type in Scenario::performTask");
  }
}

void Scenario::computeYinto(std::vector<double>& y) {
  const auto& rho = getRho();

  for (size_t nai = 0; nai < W.size(); ++nai) {
    y[nai] = W[nai] + rho[nai] * (x[nai] - z[nai]);

    if (verbosity >= 6 && nai < 3) {
      std::cout << std::setprecision(17)
                << "[DBG-Y] scen=" << problemSolver.getProblemName()
                << " nai=" << nai
                << " y=" << y[nai]
                << " W=" << W[nai]
                << " rho=" << rho[nai]
                << " x=" << x[nai]
                << " z=" << z[nai]
                << " residual=" << (y[nai] - (W[nai] + rho[nai]*(x[nai]-z[nai])))
                << std::endl;
    }
  }
}


objAndBnd Scenario::LagrangianSolve() {
  if (!setupWasCalled) {
    throw std::runtime_error("setup must be called for Scenario serialNumber="
                             + std::to_string(getSerialNumber()));
  }
  auto retval = problemSolver.doLagrangianSolve(x);
  // The Lagrangian minimizes over K_s, so its solution is already a point
  // of K_s: the two roles coincide and the export is just x.  This also
  // covers iteration 0 when --iter0-fw-step is off, where no FW solve has
  // run yet and xExport would otherwise be untouched.
  xExport = x;
  return retval;
}

void Scenario::FWsolve() {
  TicToc fwTimer; fwTimer.tic();
  problemSolver.doFWsolve(x, xExport);
  lastFWSolveTime_   = fwTimer.toc();
  lastFWNLTime_      = problemSolver.getLastFWNLTime();
  lastFWMILPTime_    = problemSolver.getLastFWMILPTime();
  lastFWMILPSolves_  = problemSolver.getLastFWMILPSolves();
}

void Scenario::_csvGuts(const std::string &dirName,
                        const std::string &fileName,
                        const std::vector<double> &vec) const {
  fs::path dir(dirName);
  fs::path fullPath = dir / fileName;
  std::ofstream file(fullPath);
  if (file.is_open()) {
    vecToOpenCSVFile(problemSolver.getNonantNames(), file);
    vecToOpenCSVFile(vec, file);
    file.close();
  } else {
    std::cerr << "Unable to open file: " << fullPath << std::endl;
  }
}

void Scenario::xToCsv(const std::string &dirName) const {
  // The end-of-run dump is the answer being reported, so it gets the exported
  // candidate (a point of K_s) rather than the algorithm's iterate.
  std::string fname = "x_" + problemSolver.getProblemName();
  _csvGuts(dirName, fname, xExport);
}

void Scenario::WToCsv(const std::string& dirName) const {
  std::string fname = "W_" + problemSolver.getProblemName();
  _csvGuts(dirName, fname, W);   // FIXED
}
