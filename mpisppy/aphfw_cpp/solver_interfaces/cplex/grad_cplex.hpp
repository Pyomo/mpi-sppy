#ifndef GRAD_CPLEX_HPP
#define GRAD_CPLEX_HPP

/*
The so-called MILP solve for FW (based on the gradient, hence "grad").
*/

#include <vector>
#include <stdexcept>
#include "cplex_interface.hpp"

class CPLEXGrad : public CPLEXInterface {
public:
  explicit CPLEXGrad(const std::string& scenarioName);
  CPLEXGrad(const std::string& scenarioName, const std::string& logFileName);  
  virtual ~CPLEXGrad() = default;
  // can by moved, but not copied
  CPLEXGrad(const CPLEXGrad&) = delete;
  CPLEXGrad& operator=(const CPLEXGrad&) = delete;
  CPLEXGrad(CPLEXGrad&&) = default;
  CPLEXGrad& operator=(CPLEXGrad&&) = default;
    
  const std::vector<double>& getGradientAtReference() const {
    return gradientAtReference;
  }
  void updateReferenceVector(const std::vector<double>& refX); // does a lot
  bool addObjectiveCutoff(double cutoffValue) override;
  void removeObjectiveCutoff() override;

private:
  std::vector<double> referenceX;
  std::vector<double> gradientAtReference;
  IloRange objCutoffConstr_;
  bool hasCutoffConstr_ = false;
};
#endif  // GRAD_CPLEX_HPP
