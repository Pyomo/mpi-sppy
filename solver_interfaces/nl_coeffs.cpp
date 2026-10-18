#include "nl_coeffs.hpp"

NLCoeffs::NLCoeffs(SolverInterface& SI,
                   std::vector<std::unique_ptr<const std::vector<double>>>& R) {
  // Note: we are trying to set the var coeffs for nonants; see convR.tex
  // TBD: this could be factored to be more efficient, but might not be worth it.
  size_t m = R.size();
  if (m == 0) {
    throw std::runtime_error("Entered NLCoeff with R.size()=0");
  }
  for (int j=0; j < m; ++j) {
    std::vector<double> rj = (*R[j]);  // typing aid
    linCoeffs[j] = 0.0;  // We will accumulate over the i indexes
    quadCoeffs[j] = 0.0;
    for (size_t i=0; i < SI.fullC.size(); ++i) {  // SI.fullC.size() is $n$
      size_t nai;  // nonant index
      if ((nai = SI.iToNai(i)) == -1) {
        linCoeffs[j] += SI.fullC[i] * rj[i];  // note: rj[i] is $r_ij$
      } else {
        linCoeffs[j] += (SI.fullC[i] + (*SI.WPtr)[nai] -SI.rho[nai]*(*SI.zPtr)[nai])* rj[i];
        quadCoeffs[j] += (SI.rho[nai]/2) * rj[i]*rj[i];
      }
    }
  }
  // Now do the cross terms for j (cross terms only for nonants)
  // as of June 2025, it is done by simulating the addition of each of the $R_k, k>1$ (one-based)
  for (size_t k = 1; k < m; ++k) {
    for (size_t j = 0; j < k; ++j) {
      crossCoeffs[{j, k}] = 0.0;
      for (size_t nai=0; nai < SI.nonantDenseIndexes.size(); ++nai) {  // only for nonants
        double rij = (*R[j])[SI.nonantDenseIndexes[nai]];
        double rik = (*R[k])[SI.nonantDenseIndexes[nai]];
        crossCoeffs[{j, k}] += SI.rho[nai] * rij * rik;
      }
    }
  }
}

