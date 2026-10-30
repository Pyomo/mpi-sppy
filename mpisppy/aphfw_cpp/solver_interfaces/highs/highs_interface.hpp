///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#ifndef HIGHS_INTERFACE_HPP
#define HIGHS_INTERFACE_HPP

/* The HiGHS backend.  HiGHS is open source, so this is the one backend that
  needs no license.

  HiGHS solves MIPs and continuous convex QPs, but not MIQPs.  The FW-APH
  solves need an MIQP only when the model's own objective is quadratic: the
  Lagrangian and the FW gradient (LMO) solves are MIPs with a linear objective,
  and the NL (prox) solve is a continuous QP.  So this backend refuses a model
  whose objective has quadratic terms, and a solve that would put a quadratic
  objective on a model with integer variables (enableLagrangianProxAt).

  HiGHS's Hessian convention is c'x + (1/2) x'Qx, so the prox term
  (rho/2) x^2 enters Q's diagonal as rho.

  Solution values are copied out of HiGHS after each successful solve and
  served from that copy: HiGHS discards its solution when the model is
  modified, and some callers read a solution after a modification.  For the
  same reason a MIP start is handed to HiGHS just before the solve (see
  setStartAtCurrent), since the cost and cutoff-row changes made between
  setting a start and solving would discard it.

  The log file is appended to through HiGHS's logging callback rather than its
  log_file option, which truncates the file when it opens it; there are three
  of these objects per scenario and they share one file.
 */

#include <Highs.h>
#include <atomic>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <iostream>
#include <unordered_set>
#include "../solver_interface.hpp"
#include "../json.hpp"
#include "../../utils/utils.hpp"

using NonantTreeIndexMap = std::map<std::string, std::vector<int>>;

//===============================================

class HiGHSInterface : public SolverInterface {
public:
  HiGHSInterface(const std::string& scenarioName, const std::string& logFileName);
  virtual ~HiGHSInterface() override = default;
  // can be moved, but not copied
  HiGHSInterface(const HiGHSInterface&) = delete;
  HiGHSInterface& operator=(const HiGHSInterface&) = delete;
  HiGHSInterface(HiGHSInterface&&) = default;
  HiGHSInterface& operator=(HiGHSInterface&&) = default;

  objAndBnd solve() override;
  void interruptSolve() override;
  std::string getVarName(const size_t i) const override {
    return highs_->getLp().col_names_.at(i);
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

  void setLogFile(const std::string fileName) override {
    std::cerr << "Warning: The log file cannot be set dynamically for HiGHS"
              << std::endl;
  }

  void setHiGHSOption(const std::string& name, double value);
  // HiGHS runs every solve in a process on one thread pool, started by the
  // first solve, and a solve asking for another size is an error.  So the
  // thread requests of all the objects are combined into one: the largest
  // made before the first solve.  A larger one made later is ignored, with a
  // warning.
  static void requestThreads(int threads);

  // from here down are functions the caller should rarely, if ever, need.
  void printSolution() override;
  void nonantSolutionVectorToCsv(std::string fileName) const override;
  void exportModel(const std::string fname) override;  // .mps or .lp suffix
  // Marks the last solution as the start for the next MIP solve.
  bool setStartAtCurrent() override;

protected:
  // Pointers, so that a move keeps the address the callback was given.
  std::unique_ptr<Highs> highs_;
  struct CallbackState {
    std::atomic<bool> stopRequested{false};
    std::ofstream log;   // not open when there is no log file
    std::mutex logMutex;
  };
  std::unique_ptr<CallbackState> cbState_;

  int numVars;      // x vars only; HiGHSNL appends its g vars after these
  std::vector<double> solution_;  // all columns, from the last good solve
  bool startPending_ = false;     // hand solution_ to the next MIP solve

  static std::atomic<int> processThreads_;      // 0: HiGHS's default
  static std::atomic<bool> threadPoolStarted_;

  void highsCheck(HighsStatus status, const std::string& what) const;
  bool hasIntegerCols() const;
  void _buildNonantsMaps();
  void clearModel() override;

private:
  static void callback_(int callbackType, const std::string& message,
                        const HighsCallbackOutput* dataOut,
                        HighsCallbackInput* dataIn, void* userData);
};
#endif // HIGHS_INTERFACE_HPP
