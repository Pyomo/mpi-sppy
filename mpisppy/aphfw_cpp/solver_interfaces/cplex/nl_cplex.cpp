#include "nl_cplex.hpp"

void _dropAllConstraints(IloModel& model);


// delegate single-arg ctor to the 2-arg ctor (empty logfile)
CPLEXNL::CPLEXNL(const std::string& scenarioName)
  : CPLEXNL(scenarioName, std::string())
{}

// new two-arg ctor accepting logfile and delegating to CPLEXInterface(scenarioName, logFileName)
CPLEXNL::CPLEXNL(const std::string& scenarioName, const std::string& logFileName)
  : CPLEXInterface(scenarioName, logFileName),
    sumToOne_(),        // null handle
    combCons_(env),
    gvars_(env)
{
  // CPLEXInterface will create the model for scenarioName (and set logfile if non-empty)
  xSize = vars.getSize(); 
  // Note that the indexes in the nonant maps still refer to x values because g is appended
  _dropAllConstraints(model);

  // relax integrality (make original x vars continuous)
  IloConversion relax(env, vars, ILOFLOAT);
  model.add(relax);                         // activate conversion
}


// add a new r and update the NL model (append terms)
void CPLEXNL::addANewr(const std::vector<double>& r) {
  if (static_cast<IloInt>(r.size()) != xSize)
    throw std::runtime_error("addANewr: r.size() != xSize");

  appendR(r);  // track in R so get_R().size() reflects actual columns added

  IloEnv env = model.getEnv();

  // new g in [0,1]
  std::string vname = "g" + std::to_string(gvars_.getSize() + 1);
  IloNumVar newg(env, 0.0, 1.0, ILOFLOAT, vname.c_str());
  model.add(newg);
  gvars_.add(newg);

  try {
    // first time: build rows
    if (!sumToOne_.getImpl()) {
      sumToOne_ = (newg == 1.0);
      sumToOne_.setName("sum_to_one");
      model.add(sumToOne_);

      combCons_.clear();
      combCons_ = IloRangeArray(env, xSize);
      for (IloInt i = 0; i < xSize; ++i) {
        IloExpr e(env);
        e = vars[i] - r[static_cast<size_t>(i)] * newg;
        combCons_[i] = (e == 0.0);
        model.add(combCons_[i]);
        e.end();
      }

    } else {
      // subsequent calls: add coefficients for the new g
      sumToOne_.setLinearCoef(newg, 1.0);
      for (IloInt i = 0; i < xSize; ++i) {
        combCons_[i].setLinearCoef(newg, -r[static_cast<size_t>(i)]);
      }
    }
  } catch (const IloException& e) {
    std::cerr << "CPLEX/Concert exception: " << e.getMessage() << std::endl;
    throw;
  }
  // TBD: Is the caller already updating? (probably not)
  // after adding newg to model and combCons_ construction/updates
  fullObjectiveRebuild();  
  
}

std::vector<double> CPLEXNL::copyEntireSolution() const {
  // for the original problem.
  // Result vector to store the weighted sum
  // create a vector and copy the values from the cplex object to it
  IloNumArray vals(env);
  cplex.getValues(vals, vars);
  std::vector<double> returnVector(vals.getSize());
  for (IloInt i = 0; i < xSize; ++i) {
    returnVector[i] = vals[i];
  }
  return returnVector;
}


/* start of for_new_versions_of_cplex
void _dropAllConstraints(IloModel& model) {
  IloEnv env = model.getEnv();

  IloConstraintArray cons(env);
  model.getConstraints(cons);      // collects *all* constraint types (IloRange, indicators, PWL, etc.)

  if (cons.getSize() > 0) {
    model.remove(cons);            // remove in one shot
  }
  cons.end();                      // release array storage
  // No update() call needed in Concert.
}
end of for_new_versions_of_cplex
*/

// Remove all constraints (linear, quadratic, indicators, PWL, conversions, etc.)
void _dropAllConstraints(IloModel& model) {
  IloEnv env = model.getEnv();

  // Collect first (don't remove while iterating)
  IloExtractableArray toRemove(env);
  for (IloModel::Iterator it(model); it.ok(); ++it) {
    IloExtractable ex = *it;
    if (ex.isConstraint()) {          // <-- check via IloExtractable
      toRemove.add(ex);
    }
  }

  // Bulk remove (if your compiler complains about this overload, use the for-loop below)
  if (toRemove.getSize() > 0) {
    model.remove(toRemove);
    // Fallback if needed:
    // for (IloInt i = 0; i < toRemove.getSize(); ++i) model.remove(toRemove[i]);
  }
  toRemove.end();
  // No update() needed in Concert.
}

