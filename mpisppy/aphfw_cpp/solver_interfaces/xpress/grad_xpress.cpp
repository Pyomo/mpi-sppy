#include "grad_xpress.hpp"
#include <numeric>

XpressGrad::XpressGrad(const std::string& scenarioName,
                       const std::string& logFileName)
  : XpressInterface(scenarioName, logFileName)
{
  gradientAtReference.assign(numVars, 0.0);
  referenceX.assign(numVars, 0.0);
}

void XpressGrad::updateReferenceVector(const std::vector<double>& refX)
// Grab the reference vector, compute the gradient, and update the objective.
// Side-effect: set current solution as warm start for the next solve.
{
  // Warm start from the current solution
  setStartAtCurrent();

  this->referenceX = refX;

  if (static_cast<int>(refX.size()) != numVars) {
    throw std::runtime_error("XpressGrad::updateReferenceVector: referenceX size mismatch");
  }
  if (!WPtr || !zPtr) {
    throw std::runtime_error("XpressGrad::updateReferenceVector: W or z not attached");
  }
  if (rho.size() != static_cast<size_t>(totalNonantCount)) {
    throw std::runtime_error("XpressGrad::updateReferenceVector: rho size unexpected");
  }

  // Compute gradient and set as objective (linearization of the full objective)
  std::vector<int> colind(numVars);
  std::iota(colind.begin(), colind.end(), 0);

  for (int i = 0; i < numVars; ++i) {
    double g = fullC[i];

    size_t nai = iToNai(i);
    if (nai != static_cast<size_t>(-1)) {
      g += (*WPtr)[nai]
           - rho[nai] * (*zPtr)[nai]
           + rho[nai] * referenceX[i];
    }

    gradientAtReference[i] = g;
  }

  // Set objective to gradient (purely linear)
  xprsCheck(XPRSchgobj(prob_, numVars, colind.data(), gradientAtReference.data()),
            "XPRSchgobj in updateReferenceVector");

  // Zero out any quadratic terms (gradient objective is linear)
  // Only need to clear terms that were previously set
  for (int j = 0; j < numVars; ++j) {
    if (fullQDiag[j] != 0.0 || varToNonant.count(j)) {
      xprsCheck(XPRSchgqobj(prob_, j, j, 0.0),
                "XPRSchgqobj(clear) in updateReferenceVector");
    }
  }
}

bool XpressGrad::addObjectiveCutoff(double cutoffValue) {
  if (cutoffRowIndex_ >= 0) removeObjectiveCutoff();
  int nrows = 0;
  xprsCheck(XPRSgetintattrib(prob_, XPRS_ROWS, &nrows), "XPRSgetintattrib(ROWS)");
  std::vector<int> colind;
  std::vector<double> rowcoef;
  for (int i = 0; i < numVars; ++i) {
    if (gradientAtReference[i] != 0.0) {
      colind.push_back(i);
      rowcoef.push_back(gradientAtReference[i]);
    }
  }
  int nnz = static_cast<int>(colind.size());
  char rowtype = 'L';
  double rhs = cutoffValue;
  int starts[] = {0};
  xprsCheck(XPRSaddrows(prob_, 1, nnz, &rowtype, &rhs, nullptr,
                         starts, colind.data(), rowcoef.data()),
            "XPRSaddrows (obj cutoff)");
  cutoffRowIndex_ = nrows;
  return true;
}

void XpressGrad::removeObjectiveCutoff() {
  if (cutoffRowIndex_ >= 0) {
    xprsCheck(XPRSdelrows(prob_, 1, &cutoffRowIndex_),
              "XPRSdelrows (obj cutoff)");
    cutoffRowIndex_ = -1;
  }
}
