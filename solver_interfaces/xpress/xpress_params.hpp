#ifndef XPRESS_PARAMS_HPP
#define XPRESS_PARAMS_HPP

#include "../solver_params.hpp"
#include "xpress_interface.hpp"

class XpressParams : public SolverParams {
public:
  explicit XpressParams(const std::string& jsonFileName);
  void pushSolverParams(SolverInterface& scenario) override;
};

#endif // XPRESS_PARAMS_HPP
