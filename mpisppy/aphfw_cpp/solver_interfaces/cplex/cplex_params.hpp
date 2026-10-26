#ifndef CPLEX_PARAMS_HPP
#define CPLEX_PARAMS_HPP

#include "../solver_params.hpp"
#include "cplex_interface.hpp"

class CPLEXParams : public SolverParams {
public:
  explicit CPLEXParams(const std::string &jsonFileName);
  void pushSolverParams(SolverInterface& scenario) override;
};

#endif // CPLEX_PARAMS_HPP
