///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "nl_gurobi.hpp"

void _dropAllConstraints(std::unique_ptr<GRBModel>& model);

GurobiNL::GurobiNL(const std::string& scenarioName,
                   const std::string& logFileName):
  GurobiInterface(scenarioName, logFileName)
{
  // Note that the indexes in the nonant maps still refer to x values because g is appended
  xSize = numVars;   // net of g vars that will be added later
  _dropAllConstraints(model);
  GRBVar* v = model->getVars();
  for (int i = 0; i < xSize; ++i) {
    v[i].set(GRB_CharAttr_VType, GRB_CONTINUOUS);
  }
  delete[] v;
}

GurobiNL::~GurobiNL() = default;

void GurobiNL::addANewr(const std::vector<double>& r) {
  // add r to R and update the NL model (i.e. append things)
  // TBD: Should we make sure we have been given W, etc?
  appendR(r);

  std::string vname = "g" + std::to_string(R.size());
  GRBVar newg = model->addVar(0, 1, 0, GRB_CONTINUOUS, vname);
  model->update();
  // TBD: Is the caller already updating? (probably not)
  // after adding newg to model and combCons_ construction/updates
  fullObjectiveRebuild();  
  // now deal with the constraints
  try {
    int numConstraints = model->get(GRB_IntAttr_NumConstrs);  
    if (numConstraints == 0) {
      // first call, so we need to add
      model->addConstr(newg == 1, "sum_to_one");  // needs to be first
      for (size_t c=1; c < xSize+1; ++c) {
        model->addConstr(vars[c-1] - r[c-1]*newg == 0);  // convex-combination constraints
                                                             }
    } else if (numConstraints == xSize + 1) {
      GRBConstr* cons = model->getConstrs();
      model->chgCoeff(cons[0], newg, 1);   // sum-to-one constraint
      for (size_t c=1; c < xSize+1; ++c) {
        // convex-combination constraints 
        model->chgCoeff(cons[c], newg, -r[c-1]);
      }
    } else {
      throw std::runtime_error("wrong number of NL constraints=" + std::to_string(numConstraints));
    }
    model->update();
  } catch (GRBException& e) {
    std::cerr << "Gurobi Error code = " << e.getErrorCode() << std::endl;
    std::cerr << "Gurobi Error message: " << e.getMessage() << std::endl;
  } catch (...) {
    std::cerr << "An unexpected error occurred during model construction or optimization." << std::endl;
  }  
  model->update();
}


void _dropAllConstraints(std::unique_ptr<GRBModel>& model) {
  // Linear constraints
  int n = model->get(GRB_IntAttr_NumConstrs);
  if (n > 0) {
    GRBConstr* c = model->getConstrs();
    for (int i = 0; i < n; ++i) model->remove(c[i]);
    delete[] c;
  }

  // Quadratic constraints
  int nq = model->get(GRB_IntAttr_NumQConstrs);
  if (nq > 0) {
    GRBQConstr* qc = model->getQConstrs();
    for (int i = 0; i < nq; ++i) model->remove(qc[i]);
    delete[] qc;
  }
  // SOS constraints
  int ns = model->get(GRB_IntAttr_NumSOS);
  if (ns > 0) {
    GRBSOS* sos = model->getSOSs();
    for (int i = 0; i < ns; ++i) model->remove(sos[i]);
    delete[] sos;
  }

  // General constraints (indicators, PWL, etc.)
  int ng = model->get(GRB_IntAttr_NumGenConstrs);
  if (ng > 0) {
    GRBGenConstr* gc = model->getGenConstrs();
    for (int i = 0; i < ng; ++i) model->remove(gc[i]);
    delete[] gc;
  }

  model->update();  // apply deletions
}


std::vector<double> GurobiNL::copyEntireSolution() const {
  // for the original problem.
  // create a vector and copy the values from the gurobi object to it
  std::vector<double> returnVector(numVars);
  for (int i = 0; i < xSize; ++i) {
    returnVector[i] = vars[i].get(GRB_DoubleAttr_X);
  }
  return returnVector;
}
