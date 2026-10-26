#ifndef PARAMS_HPP
#define PARAMS_HPP

#include "params.hpp"
#include "solver_interface.hpp"  

/* Header-only, base class for supported solver parameters; 
   derive a class for a given solver.
Note: we allow getting and setting supported params, so we disable public
      access to the original map from the json file.
For free-form additional parameters, add support in the derived class.
 */
// caller has: if (auto gap = foo.getMipGap())

class SolverParams : public ParamsBase {
public:
  SolverParams(const std::string jsonFileName):
    ParamsBase(jsonFileName)
  {
    if (paramMap.count("mipGap")) {
      mipGap = std::get<double>(paramMap["mipGap"]);
    }
    if (paramMap.count("timeLimit")) {
      timeLimit = std::get<double>(paramMap["timeLimit"]);
    }
    if (paramMap.count("threads"))
      threads = std::get<int>(paramMap["threads"]);
  }
  virtual ~SolverParams() { }
  // A derived class will set supported  parameters in the solver
  virtual void pushSolverParams(SolverInterface& scenario) = 0;
  std::optional<double> getMipGap() const {
    return mipGap;
  }
  std::optional<double> getTimeLimit() const {
    return timeLimit;
  }
  std::optional<int> getThreads() const {
    return threads;
  }
  void setMipGap(double v) {
    mipGap = v;
  }
  void setTimeLimit(double v) {
    timeLimit = v;
  }
  void setThreads(int v) {
    threads = v;
  }
  // set for error checking by derived classes, if desired
  std::set<std::string> canHave = {"mipGap", "timeLimit", "threads"};

protected:
  std::optional<double> mipGap;
  std::optional<double> timeLimit;
  std::optional<int> threads;

private:
  // map changes not supported
  ParamMapType& getParamMap() override final {
    throw std::runtime_error("getParamMap is not supported in SolverParams");
  }  
};

#endif  // ifndef
