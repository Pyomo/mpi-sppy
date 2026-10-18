#include "grad_gurobi.hpp"

GurobiGrad::GurobiGrad(const std::string& scenarioName,
                       const std::string& logFileName):
  GurobiInterface(scenarioName, logFileName)
{
  // gradient for the original problem
  gradientAtReference.assign(numVars, 0.0);

  // Keep the original objective's quadratic terms; updateReferenceVector
  // rebuilds the objective from them every call.  objExpr itself is the
  // original objective here (the base class reads it at construction and does
  // not use it afterwards), so this runs before anything has modified it.
  const unsigned int qsize = objExpr.size();
  for (unsigned int k = 0; k < qsize; ++k) {
    baseQuadObj_ += objExpr.getCoeff(k) * objExpr.getVar1(k) * objExpr.getVar2(k);
  }
}

void GurobiGrad::updateReferenceVector(const std::vector<double>& referenceX)
// grab the reference vector and update the objective function
// also update the gradient
// Another side-effect: put the current solution in as the starting solution
// TBD: Should we verify that we have been given W etc?
{
  // first, get the current solution as a starting point
  if (!setStartAtCurrent()) {
    // Having a starting solution can reduce solution time (sometimes a lot)
    std::cout << "HEY! someday we need iter0 as a  starting solution in grad gurobi" << std::endl;
  }
  //printVectorInline("fullC", fullC);
  // Rebuild the objective rather than adjust the one built last time.  Adding
  // (new - current) coefficients to a kept expression gives the same numbers,
  // but leaves every superseded term in it: the expression handed to
  // setObjective grew by 2 * numVars terms per solve, so by cycle k it carried
  // k times the terms the model has.  Starting from the quadratic part each
  // call keeps it the size of the objective, and drops a per-variable
  // attribute read as well.
  objExpr = baseQuadObj_;
  for (size_t i=0; i < fullC.size(); ++i) {
    gradientAtReference[i] = fullC[i];
    size_t nai;  // nonant index
    if ((nai = iToNai(i)) != -1) {
      gradientAtReference[i] += (*WPtr)[nai] - rho[nai] * (*zPtr)[nai] + rho[nai] * referenceX[i];
    }
    objExpr += gradientAtReference[i] * vars[i];
  }
  model->setObjective(objExpr, model->get(GRB_IntAttr_ModelSense));
  model->update();
}

bool GurobiGrad::addObjectiveCutoff(double cutoffValue) {
  // An explicit row, objective <= cutoffValue, as mpi-sppy's FWPH adds, rather
  // than the Cutoff parameter, which only prunes nodes.  As a row it takes part
  // in presolve and bound propagation, which on some subproblems shortens the
  // proof of optimality a great deal.
  if (hasCutoffConstr_) removeObjectiveCutoff();
  if (objExpr.size() == 0) {
    cutoffConstr_ = model->addConstr(objExpr.getLinExpr() <= cutoffValue, "fw_obj_cutoff");
    cutoffIsQuadratic_ = false;
  } else {
    cutoffQConstr_ = model->addQConstr(objExpr <= cutoffValue, "fw_obj_cutoff");
    cutoffIsQuadratic_ = true;
  }
  hasCutoffConstr_ = true;
  tolerateInfeasible_ = true;
  infeasibleDueToCutoff_ = false;
  return true;
}

void GurobiGrad::removeObjectiveCutoff() {
  if (hasCutoffConstr_) {
    if (cutoffIsQuadratic_) {
      model->remove(cutoffQConstr_);
    } else {
      model->remove(cutoffConstr_);
    }
    hasCutoffConstr_ = false;
  }
  tolerateInfeasible_ = false;
}
