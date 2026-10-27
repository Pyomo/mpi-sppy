///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
// originally contributed by Jonathan Eckstein

#pragma once

#include "groupController.hpp"
#include "scenario.hpp"
#include "treeNode.hpp"
#include "scenSolveType.hpp"
#include <deque>
#include <span>
#include <iomanip>
/*
  Notes:
  - Be careful about scenario numbers vs indexes (global and local):
    = the names themselves could be one-based (which doesn't matter much),
    = we have a set of local scenario indexes with values that are global.
    = we have a vector called localScenarios (so that index is local)
    = int s is always a local index !!!  (otherwise say int sGlobal)
 */


// The groupController-derived class that will be a base for APH and
// possible other algorithms

class spControllerBase :
public groupController<Scenario>
{
public:

  // Constructor (minimal)

  spControllerBase() = default;

  void setup(Config& cfg) override;

  // Set an explicit scenario assignment for this rank (call before setup).
  // When set, setup() uses these names instead of evenSplit.
  void setExplicitScenarioNames(std::vector<std::string> names) {
    explicitScenarioNames_ = std::move(names);
  }

  // access functions
  std::vector<int>& getLocalScenarioIndexList() {return localScenarioIndexList; };
  const std::string& getProblemName() const {return problemName; }
  // writeable view of the scenarios on this rank (int numSubdomains is uncool...)
  std::span<Scenario> localScenarios() {
    return { subdomainObj, static_cast<std::size_t>(numSubdomains) };
  }
  // read-only view (for const objects/contexts)
  std::span<const Scenario> localScenarios() const {
    return { subdomainObj, static_cast<std::size_t>(numSubdomains) };
  }  
  
  Scenario& scenario(const int s) {return localScenarios()[s]; };


  // More readable name of scenario objects

  Scenario& scenObject(const int i) { return subdomainObj[i]; };

  // check the data structures
  void checkTreeConsistency();  

  // Destructor

  virtual ~spControllerBase() { };


protected :


  // Compute a Lagranangian bound outside the main event loop

  double lagrangianBound();


  //// Data members

  // Scenario tree information (local to this rank)

  std::vector<double> scenProb;         // Scenario probabilities

  std::vector<ScenarioTreeNode> localNodes;  // Local tree nodes (everything above leaves/scenarios)

  // list of scenarios on this rank
  std::vector<int> localScenarioIndexList;

  // What problem are we working on (a directory name)
  std::string problemName;

  // global scenario cout
  int globalScenCount = 0;

  // Optional explicit scenario names (set via setExplicitScenarioNames)
  std::vector<std::string> explicitScenarioNames_;
};

