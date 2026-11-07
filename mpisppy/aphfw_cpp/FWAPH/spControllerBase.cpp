///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "spControllerBase.hpp"
#include "utils/evenSplit.hpp"
#include "utils/utils.hpp"
#include <algorithm>
#include <unordered_map>

/*
  Note that using the scheme with no constructor args followed by setup is very fragile and order matters.
 */
void spControllerBase::setup(Config& cfg) {

  // It is a little uncool that we are requiring problem-name to be put in cfg elsewhere
  problemName = cfg.getRequiredString("problem-name");

  // Full paths without extension (e.g., "/tmp/dir/Scenario0"), indexed by
  // the entries of localScenarioIndexList.
  std::vector<std::string> scenarioPaths;

  if (!explicitScenarioNames_.empty()) {
    // Use explicit scenario assignment (e.g., from mpi-sppy hub wrapper).  The
    // directory need hold only this rank's scenarios, so each scenario is on
    // one rank and the ranks' counts add up to the global count.
    numSubdomains = (int)explicitScenarioNames_.size();
    MPI_Allreduce(&numSubdomains, &globalScenCount, 1, MPI_INT, MPI_SUM, comm);

    localScenarioIndexList.resize(numSubdomains);
    std::iota(localScenarioIndexList.begin(), localScenarioIndexList.end(), 0);
    for (const auto& name : explicitScenarioNames_) {
      // Explicit names from mpi-sppy are just stems ("Scenario0"); strip any path.
      const fs::path path = fs::path(problemName) / fs::path(name).filename();
      if (!fs::exists(fs::path(path.string() + ".mps")))
        throw std::runtime_error("Explicit scenario name '" + name +
                                 "' not found in problem directory " + problemName);
      scenarioPaths.push_back(path.string());
    }
    // verbosity is not set until groupController::setup below, so read it here
    if (cfg.getOptionValue<int>("verbosity") >= 1)
      std::cerr << "globalScenCount=" << globalScenCount
                << " numSubdomains=" << numSubdomains
                << " (explicit assignment)" << std::endl;
  } else {
    scenarioPaths = getLPFilesWithoutExtension(problemName);
    globalScenCount = (int)scenarioPaths.size();
    if (globalScenCount == 0) {
      throw std::runtime_error("No scenario lp files found for problemName=" + problemName);
    }

    // Determine which scenarios are on this rank
    auto splitter = evenSplit();
    splitter.setup(globalScenCount, comm);
    numSubdomains = splitter.myNumItems();  // needs to be set before groupContoller::setup is called
    std::cerr << "globalScenCount=" << globalScenCount << " numSubdomains=" << numSubdomains << std::endl;

    localScenarioIndexList.resize(numSubdomains);
    std::iota(localScenarioIndexList.begin(),
              localScenarioIndexList.end(),
              splitter.myStartPos());
  }

  // Needs to be called
  groupController::setup(cfg);  // there is a fragile connection via subdomainObj (Obj == Objects)

  // Set up the scenarios themselves as well as scenProb.
  // While doing so, grab globalNodeCount ASAP and reserve localNodes to avoid
  // std::vector reallocation invalidating node pointers stored in nodePtrMap.
  int globalNodeCount = 0;
  bool haveGlobalNodeCount = false;

  scenProb.resize(localScenarioIndexList.size());
  for (int lsi = 0; lsi < (int)localScenarioIndexList.size(); ++lsi) {
    auto si = localScenarioIndexList[lsi];
    auto& scen = localScenarios()[lsi];

    scen.setup(cfg, scenarioPaths[si]);
    scenProb[lsi] = scen.getProbability();

    if (!haveGlobalNodeCount) {
      globalNodeCount = scen.getGlobalNodeCount();
      haveGlobalNodeCount = true;
      if (globalNodeCount < 0) globalNodeCount = 0;
      localNodes.clear();
      localNodes.reserve((size_t)globalNodeCount);  // <-- fixes pointer invalidation during emplace_back
    }
  }
  if (!haveGlobalNodeCount) {
    globalNodeCount = 0;
  }

  // We will now create the nodes containing local scenarios; then make their comms.

  // this map goes from global serial number to local index
  std::unordered_map<size_t, size_t> serialNumToLIndexMap;

  // this map goes from node name to a node pointer (to find parents)
  std::map<std::string, ScenarioTreeNode*> nodePtrMap;

  // We will loop over local scenarios and process nodes as we encounter them.
  for (int localScenI = 0; localScenI < (int)localScenarios().size(); ++localScenI) {
    auto &scen = localScenarios()[localScenI];

    for (const auto& [ndName, ndData] : scen.getTreeNodeDataMap()) {

      if (!serialNumToLIndexMap.contains(ndData.serialNumber)) {
        // If we are in this scope, we need to create the node, etc.
        auto idx = localNodes.size();                   // The index of the new node
        auto& node = localNodes.emplace_back();

        ScenarioTreeNode* parentPtr = nullptr;   // leave it null for "ROOT"
        if (ndName != "ROOT") {
          std::string parentName = strip_suffix(ndName);
          auto it = nodePtrMap.find(parentName);
          if (it == nodePtrMap.end() || it->second == nullptr) {
            throw std::runtime_error("Parent node not found for " + ndName + " parent=" + parentName);
          }
          parentPtr = it->second;
        }

        nodePtrMap[ndName] = &node;               // safe now because reserve prevents reallocation
        node.setup(ndName, ndData, scen, parentPtr);
        node.addLocalScenIndex(localScenI);
        serialNumToLIndexMap.emplace(ndData.serialNumber, idx);

      } else {
        // We don't need to add the node, but we do need to add the scenario to its node.
        auto& node = localNodes[serialNumToLIndexMap.at(ndData.serialNumber)];
        node.addLocalScenIndex(localScenI);
      }
    }
  }

  // Make the comms via splits
  // Now the loop to make the comms (every rank will do the same loop)
  for (int globalNodeNum = 0; globalNodeNum < globalNodeCount; ++globalNodeNum) {
    MPI_Comm tmp = MPI_COMM_NULL;
    int color = serialNumToLIndexMap.contains((size_t)globalNodeNum) ? globalNodeNum : MPI_UNDEFINED;

    MPI_Comm_split(comm, color, myRank, &tmp);

    if (tmp != MPI_COMM_NULL) {
      std::string nm = "comm_for_node_" + std::to_string(globalNodeNum);
      MPI_Comm_set_name(tmp, nm.c_str());
      localNodes[serialNumToLIndexMap.at((size_t)globalNodeNum)].setComm(tmp);
    }
  }



  setMaxConcurrentComms(globalNodeCount);  // in groupController
}
