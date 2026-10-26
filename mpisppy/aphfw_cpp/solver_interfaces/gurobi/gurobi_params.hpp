#ifndef GUROBI_PARAMS_HPP
#define GUROBI_PARAMS_HPP

#include "../solver_params.hpp"
#include "gurobi_interface.hpp"

class GurobiParams : public SolverParams {
public:
  GurobiParams(const std::string jsonFileName);
  void pushSolverParams(SolverInterface& scenario) override;
};

#endif
