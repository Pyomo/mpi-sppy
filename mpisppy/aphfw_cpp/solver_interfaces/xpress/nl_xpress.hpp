#ifndef NL_XPRESS_HPP
#define NL_XPRESS_HPP

/*
The so-called NL solve for FW (Xpress version).
We get a set of points, R. Starting with one point and growing one at a time.

The optimization vars are x and g (weight variables) with dimension |R| and
the only constraints are that g sums to one and x is the convex combination:
  sum_{r in R} g_r = 1
  x_i - sum_{r in R} r_i * g_r = 0

The constructor loads the model, drops all constraints, and relaxes integrality.
Then addANewr incrementally builds the NL model.
*/

#include "xpress_interface.hpp"
#include <numeric>
#include <algorithm>

class XpressNL : public XpressInterface {
public:
  XpressNL(const std::string& scenarioName, const std::string& logFileName);
  ~XpressNL() override = default;

  // move-only
  XpressNL(const XpressNL&) = delete;
  XpressNL& operator=(const XpressNL&) = delete;
  XpressNL(XpressNL&&) = default;
  XpressNL& operator=(XpressNL&&) = default;

  void addANewr(const std::vector<double>& r);
  std::vector<double> copyEntireSolution() const override;

  std::vector<std::unique_ptr<const std::vector<double>>>& get_R() { return R; }

private:
  std::vector<std::unique_ptr<const std::vector<double>>> R;
  void appendR(const std::vector<double>& r) {
    R.push_back(std::make_unique<const std::vector<double>>(r));
  }
  int xSize = 0;        // number of original x variables (before g vars)
  int numRows_ = 0;     // number of rows after first addANewr (1 + xSize)

  void dropAllConstraints();
  void relaxIntegrality();
};

#endif  // NL_XPRESS_HPP
