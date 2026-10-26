#ifndef FW_HPP
#define FW_HPP

#include <atomic>
#include <limits>
#include <numeric>
#include <vector>
#include "../utils/utils.hpp"
#include "../config/config.hpp"
#include "../utils/timer.hpp"
#include "nl_coeffs.hpp"

#define STRINGIFY(x) #x

/* The Frank Wolfe (1956) algorithm class.
Marguerite Straus Frank  died Dec 11, 2024 in Palo Alto at the age of 97 (her late husband had been a
professor at Stanford). There are some celebrations of her career by Anna Nagurny on the web.
Meanwhile, Roger's office at IBM was across the hall from Phillip Wolfe and they were very friendly.
   
The caller needs to create the two classes that are used
and give them solver options, W, rho, etc. The caller needs to give the initial
member of R when they call go.
We are going to call the gradient object MILP because Jonathan does,
even though I don't like that name.

I said this elsewhere, but TBD: think even more about who owns what.
 */

// header-only because it is a template class
// NLT: eg CPLEXNL; GradT: e.g. CPLEXGrad
// Note that tMax and tau are passed in to the go functions (the cfg versions are not used here)

template <typename NLT, typename GradT>
class FrankWolfe {
public:
  FrankWolfe(NLT& NL,
             GradT& MILP,
             Config& cfg):
    NL(NL),
    MILP(MILP),
    finalNonantSolution(NL.getTotalNonantCount(), 0.0),
    finalVertexSolution(NL.getTotalNonantCount(), 0.0),
    cfg(cfg),
    useFWObjCutoff_(cfg.getOptionValue<bool>("FW-obj-cutoff")),
    fwObjCutoffTol_(cfg.getOptionValue<double>("FW-obj-cutoff-tol")),
    exportPolicy_(parseExportPolicy(
                    cfg.getOptionValue<std::string>("FW-export-policy")))
  {
  }

  // What goes in finalVertexSolution, which is the candidate solution handed
  // to the xhat spokes and the denouement.  finalNonantSolution is always the
  // proximal minimizer and is not governed by this.
  enum class ExportPolicy {
    bestColumn,  // the column of R minimizing the current subproblem objective
    lastVertex,  // the newest LMO answer (what every exit used to give at tau)
    prox         // export the proximal minimizer too; for Fejer studies
  };

  static ExportPolicy parseExportPolicy(const std::string& name) {
    if (name == "best-column") return ExportPolicy::bestColumn;
    if (name == "last-vertex") return ExportPolicy::lastVertex;
    if (name == "prox")        return ExportPolicy::prox;
    throw std::runtime_error("FW-export-policy must be best-column, "
                             "last-vertex or prox; got '" + name + "'");
  }
  ~FrankWolfe() {};


  // Expose R to callers (note that this is non-const)
  std::vector<std::unique_ptr<const std::vector<double>>>& get_R() {return NL.get_R();}

  // Subproblem objective of the column exported by the most recent FW solve,
  // evaluated with the W, z and rho that were in force when the choice was
  // made.  That is exactly the quantity best-column minimizes, so it is the
  // only vantage point from which the policies can be compared: by the time a
  // cycle ends, W and z have moved.  NaN when no column was chosen -- the prox
  // policy, or before the first column is adjoined.
  double getLastExportObjVal() const { return lastExportObjVal_; }

  // Cooperative abort: when the attached flag goes up, goBody stops between
  // solves (in-flight solves are interrupted separately via the solver
  // interfaces) and finalNonantSolution is left as-is for the caller to discard.
  void setAbortFlag(const std::atomic<bool>* p) { abortFlagPtr_ = p; }

  
  void go(const std::vector<double>& r, const int tMax, const double tau) {
    // for PH and APH applications, this will only be called once.
    NL.addANewr(r);
    auto refVal = MILP.computeObjFctVal(r);
    _vb(cfg, "entered go in fw.hpp and added a new r with obj value=" + std::to_string(refVal));
    goBody(tMax, tau);
  }
  
  void goAgain(const int tMax, const double tau) {
    // Presumably, data was updated since the last call.
    if (_vb(cfg, 5)) {
      std::cout << "in goAgain: NL.pprint()=" << std::endl;
      NL.pprint(std::cout);
    }
    goBody(tMax, tau);
  }

  
  void goBody(const int tMax, const double tau) {

    // ---- Local helpers to hide the debug blocks ----
    auto debugNL = [&](const std::vector<double>& nlSolution,
                       const std::string& sname) {
      if (!_vb(cfg, 8)) return;

      std::string bname = "NL" + std::to_string(itersSoFar) + "_" +
        sname.substr(sname.size() - 5);
      std::cout << "debug NL output in fw.hpp for " << bname << std::endl;
      NL.exportModel(bname + ".lp");
      write_vector_to_csv(nlSolution, bname + ".soln");
    };

    auto debugMILP = [&](const std::vector<double>& nlSolution,
                         const std::vector<double>& milpSolution,
                         const std::vector<double>& gradient,
                         const std::vector<double>& diff,
                         const std::string& sname) {
      if (!_vb(cfg, 8)) return;

      auto bname = "MILP" + std::to_string(itersSoFar) + "_" +
        sname.substr(sname.size() - 5);
      MILP.exportModel(bname + ".lp");
      write_vector_to_csv(milpSolution, bname + ".soln");
      write_vector_to_csv(gradient,   bname + ".grad");
      write_vector_to_csv(diff,       bname + ".diff");
      auto objVal = MILP.computeObjFctVal(milpSolution);
      auto refVal = MILP.computeObjFctVal(nlSolution);
      if (_vb(cfg,5)) {
        std::cout << "The reference solution obj= " << refVal
                  << " while the latest MILP solution obj=" << objVal
                  << std::endl;
      }
    };
    
    // ----------------- start of the actual function -------------
   // Per-call MILP counter and timing accumulators
   MILPSolvesSoFar = 0;
   accumulatedNLTime_ = 0.0;
   accumulatedMILPTime_ = 0.0;

   while (true) {
     if (abortRequested_()) break;

     // --- iteration (global) counter: increment on *every* FW iteration
     itersSoFar++;

     // 1. Solve NL at current r
     {
       TicToc nlTimer; nlTimer.tic();
       NL.solve();
       accumulatedNLTime_ += nlTimer.toc();
     }
     if (NL.wasInterrupted()) break;  // no extraction; results are discarded
     if (NL.hasNumericalBailout()) {
       NL.extractNonantSolution(finalNonantSolution);
       setVertexExport(nullptr);
       break;
     }
     auto nlSolution = NL.copyEntireSolution();
     auto sname      = NL.getScenarioName();

     debugNL(nlSolution, sname);

     // 2. Enforce tMax: if we've already done tMax MILP solves, stop now
     //    (tMax is a limit on MILP solves per goBody call)
     if (MILPSolvesSoFar >= tMax) {
       if (_vb(cfg, 1))
         global_toc("FW terminates on tMax=" + std::to_string(tMax) +
                    " with MILPSolvesSoFar=" + std::to_string(MILPSolvesSoFar));
       NL.extractNonantSolution(finalNonantSolution);
       setVertexExport(nullptr);
       break;
     }

     // 3. Build gradient model around the NL solution and solve MILP
     MILP.updateReferenceVector(nlSolution);

     // Optional: add objective cutoff to force improving direction, as
     // mpi-sppy's FWPH does (_add_objective_cutoff in mpisppy/opt/fwph.py).
     bool cutoffActive = false;
     if (useFWObjCutoff_) {
       // nlSolution may include NL-specific vars (g-weights); truncate to MILP size
       auto nlForMILP = nlSolution;
       nlForMILP.resize(MILP.get_fullC_const().size());
       // The cutoff bounds the objective the MILP minimizes, the linearization
       // at nlSolution, so it is measured with that objective and not with the
       // proximal one (computeObjFctVal), which differs from it by the prox terms.
       double cutoffVal = gradObjVal(nlForMILP);
       // Never below the value of the MIP start, which is the previous vertex
       // (updateReferenceVector sets it), so that the start stays feasible: the
       // solve then always returns a vertex, and its bound remains a bound on
       // the unrestricted problem.
       if (prevMilpSolution_.size() == nlForMILP.size()) {
         cutoffVal = std::max(cutoffVal, gradObjVal(prevMilpSolution_));
       }
       // The slack is FWPH's, not tau: tau = 0.01 put the cutoff several units
       // above the optimum of an objective near -400, where it no longer
       // shortens the proof of optimality.
       double epsilon = std::max(fwObjCutoffTol_, std::abs(fwObjCutoffTol_ * cutoffVal));
       if (epsilon == 0.0) epsilon = 1e-6;  // floor to avoid degenerate cutoff
       cutoffActive = MILP.addObjectiveCutoff(cutoffVal + epsilon);
     }

     TicToc milpTimer_; milpTimer_.tic();
     auto sRes = MILP.solve();
     accumulatedMILPTime_ += milpTimer_.toc();
     MILPSolvesSoFar++;

     if (MILP.wasInterrupted()) {
       if (cutoffActive) {
         MILP.removeObjectiveCutoff();  // keep the model clean for reuse
       }
       break;  // no extraction; results are discarded
     }

     // Always remove cutoff constraint after solve (whether feasible or not)
     if (cutoffActive) {
       MILP.removeObjectiveCutoff();
       if (MILP.isInfeasibleDueToCutoff()) {
         // No improving direction found; terminate FW with current NL solution
         _vb(cfg, "FW: no improving MIP direction (cutoff infeasible); using NL solution");
         NL.extractNonantSolution(finalNonantSolution);
         setVertexExport(nullptr);
         break;
       }
     }

     if (MILP.hasNumericalBailout()) {
       NL.extractNonantSolution(finalNonantSolution);
       setVertexExport(nullptr);
       break;
     }
     auto milpSolution = MILP.copyEntireSolution();
     prevMilpSolution_ = milpSolution;
     const std::vector<double>& gradient = MILP.getGradientAtReference();

     // 4. Compute FW gap
     std::vector<double> diff(nlSolution.size());
     std::transform(nlSolution.begin(), nlSolution.end(),
                    milpSolution.begin(), diff.begin(),
                    [](double a, double b) { return a - b; });

     // the trick is that we need the actual mipgap to add to the inner product
     double mg = sRes.obj - sRes.bnd;   // assume minimizing
     double FWgap = mg + std::inner_product(
                              gradient.begin(), gradient.end(), diff.begin(), 0.0
                                            );

     if (_vb(cfg, 1))
       global_toc("FWgap=" + std::to_string(FWgap) +
                  " MILPSolvesSoFar=" + std::to_string(MILPSolvesSoFar));

     debugMILP(nlSolution, milpSolution, gradient, diff, sname);

     // 5. Check tau stopping: if gap is small, we’re done
     if (FWgap <= tau) {
       if (_vb(cfg, 1))
         global_toc("FW terminates on tau=" + std::to_string(tau) +
                    " FWGap=" + std::to_string(FWgap) +
                    " MILPSolvesSoFar=" + std::to_string(MILPSolvesSoFar));
       // The algorithm gets the proximal minimizer whose gap was just shown
       // small, not the vertex that showed it: the pair (x_s, y_s) handed to
       // the projection has to lie in the graph of the subdifferential.  The
       // vertex goes to the export instead.
       NL.extractNonantSolution(finalNonantSolution);
       setVertexExport(&milpSolution);
       break;
     }

     // 6. Otherwise add the new FW point and continue
     NL.addANewr(milpSolution);
     // (loop continues; next pass will bump itersSoFar again)
   }
  }


  std::vector<double>& getFinalNonantSolution() { return finalNonantSolution; }
  std::vector<double>& getFinalVertexSolution() { return finalVertexSolution; }
  int getItersSoFar() const { return itersSoFar; }
  int getMIPLSolvesSoFar() const { return MILPSolvesSoFar; }
  int getMILPSolvesSoFar() const { return MILPSolvesSoFar; }
  double getAccumulatedNLTime()   const { return accumulatedNLTime_; }
  double getAccumulatedMILPTime() const { return accumulatedMILPTime_; }
  void extractNodeSolution(const std::string nodeName,
                           std::vector<double>& nodeSolution) const {
    auto nI =  MILP.getNodeIndexes(nodeName);
    if ( nodeSolution.size() != nI.size() ) {
      std::cerr << "nodeSolution.size()=" << nodeSolution.size()
                << " nonantIndex[nodeName].size()=" << nI.size()
                << std::endl;
      throw std::runtime_error("bad call to extractNodeSolution");
    }
    int j = 0;
    for (const auto& i : nI) {
      nodeSolution[j++] = finalNonantSolution[i];
    }
  }
  
private:
  // Value at x of the objective the MILP minimizes after
  // updateReferenceVector: the original objective with its linear part
  // replaced by the gradient at the reference point.
  double gradObjVal(const std::vector<double>& x) const {
    const auto& c = MILP.get_fullC_const();
    const auto& g = MILP.getGradientAtReference();
    double val = MILP.computeOrigObjFctVal(x);
    for (size_t i = 0; i < x.size(); ++i) val += (g[i] - c[i]) * x[i];
    return val;
  }

  NLT& NL;
  GradT& MILP;
  Config& cfg;
  bool useFWObjCutoff_ = false;
  double fwObjCutoffTol_ = 1e-4;
  // The last vertex the MILP returned, which is its MIP start for the next solve.
  std::vector<double> prevMilpSolution_;
  const std::atomic<bool>* abortFlagPtr_ = nullptr;

  bool abortRequested_() const {
    return abortFlagPtr_ && abortFlagPtr_->load(std::memory_order_relaxed);
  }

  int itersSoFar = 0;  // not reset by go or go_again
  int MILPSolvesSoFar = 0;  // reset by go and go_again
  double accumulatedNLTime_ = 0.0;    // reset per goBody() call
  double accumulatedMILPTime_ = 0.0;  // reset per goBody() call
  std::vector<double> finalNonantSolution; // memory owned here
  std::vector<double> finalVertexSolution; // memory owned here
  ExportPolicy exportPolicy_ = ExportPolicy::bestColumn;
  double lastExportObjVal_ = std::numeric_limits<double>::quiet_NaN();

  // best-column ranks every column of R at every FW exit, and R is append-only
  // for the life of the run (doFWsolve calls go() only when R is empty, and
  // goAgain otherwise), so |R| reaches tMax * cycles and a naive rescan costs
  // O(|R| * numVars) per scenario per cycle.  A column's *original* objective
  // never changes, so it is computed once, when the column is first seen; only
  // the W/z/rho part is re-evaluated, at nonant length.  Caching the full
  // subproblem value instead would buy nothing, because W moves every cycle.
  std::vector<double> colOrigVal_;                     // parallel to NL.get_R()
  std::vector<const std::vector<double>*> colCached_;  // what colOrigVal_ is of

  // Subproblem objective of R[i], with the original part cached.  The stored
  // pointer is only an identity check: on any mismatch the entry and every
  // entry after it are discarded and refilled, so a column set that is rebuilt
  // rather than appended to still gets correct values.
  double columnObjVal_(size_t i, const std::vector<double>& col) {
    if (i >= colCached_.size() || colCached_[i] != &col) {
      colCached_.resize(i + 1, nullptr);
      colOrigVal_.resize(i + 1, 0.0);
      colCached_[i] = &col;
      colOrigVal_[i] = MILP.computeOrigObjFctVal(col);
    }
    return colOrigVal_[i] + MILP.computeNonantPenalty(col);
  }

  // Copy the nonant entries of a full-length column into a nonant vector.
  // nonantDenseIndexes is built by concatenating nonantIndexes in map order,
  // which is the order extractNonantSolution writes in, so this lands in the
  // same layout the export has always used.
  void projectColumnToNonants(const std::vector<double>& column,
                              std::vector<double>& out) const {
    const auto& dense = MILP.getNonantDenseIndexesConst();
    if (out.size() != dense.size()) {
      throw std::runtime_error("projectColumnToNonants: out.size()=" +
                               std::to_string(out.size()) + " dense.size()=" +
                               std::to_string(dense.size()));
    }
    for (size_t j = 0; j < dense.size(); ++j) {
      const int vidx = dense[j];
      if (vidx < 0 || (size_t) vidx >= column.size()) {
        throw std::runtime_error("projectColumnToNonants: column too short for "
                                 "nonantDenseIndexes[" + std::to_string(j) +
                                 "]=" + std::to_string(vidx));
      }
      out[j] = column[vidx];
    }
  }

  const std::vector<double>* newestColumn() const {
    auto& R = NL.get_R();
    return R.empty() ? nullptr : R.back().get();
  }

  // Fill finalVertexSolution.  freshVertex is this pass's LMO answer when
  // there is one -- it is not yet in R -- and nullptr at the exits that leave
  // without a usable MILP solve.
  void setVertexExport(const std::vector<double>* freshVertex) {
    lastExportObjVal_ = std::numeric_limits<double>::quiet_NaN();

    if (exportPolicy_ == ExportPolicy::prox) {
      // Nonant length, not full length, so the subproblem objective does not
      // apply and lastExportObjVal_ stays NaN.
      finalVertexSolution = finalNonantSolution;
      return;
    }

    const std::vector<double>* chosen = nullptr;
    double chosenVal = std::numeric_limits<double>::quiet_NaN();

    if (exportPolicy_ == ExportPolicy::lastVertex) {
      chosen = freshVertex ? freshVertex : newestColumn();
      if (chosen != nullptr) chosenVal = MILP.computeObjFctVal(*chosen);
    } else {
      // best-column: the subproblem objective evaluated at every column.  No
      // extra solve, and no exit-specific special case.  computeOrigObjFctVal
      // throws on a length mismatch, and go() has already evaluated the first
      // column, so a bad column set fails on the first call rather than here.
      //
      // Note that last-vertex's pick is always in this candidate set --
      // freshVertex when there is one, R.back() otherwise -- so best-column's
      // value is no worse, by construction rather than by luck of a fixture.
      double bestVal = 0.0;
      if (freshVertex != nullptr) {
        chosen = freshVertex;
        bestVal = MILP.computeObjFctVal(*freshVertex);
      }
      auto& R = NL.get_R();
      for (size_t i = 0; i < R.size(); ++i) {
        const std::vector<double>* c = R[i].get();
        if (c == nullptr) continue;
        const double v = columnObjVal_(i, *c);
        if (chosen == nullptr || v < bestVal) { chosen = c; bestVal = v; }
      }
      chosenVal = bestVal;
    }

    if (chosen == nullptr) {
      // Nothing to choose from; the proximal minimizer is the only point we
      // have.  Only reachable before the first column is adjoined.
      finalVertexSolution = finalNonantSolution;
    } else {
      projectColumnToNonants(*chosen, finalVertexSolution);
      lastExportObjVal_ = chosenVal;
    }
  }
};

#endif  // FW_HPP
