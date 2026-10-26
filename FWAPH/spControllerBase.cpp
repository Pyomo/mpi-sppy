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

  auto globalScenarioNames = getLPFilesWithoutExtension(problemName);
  globalScenCount = (int)globalScenarioNames.size();
  if (globalScenCount == 0) {
    throw std::runtime_error("No scenario lp files found for problemName=" + problemName);
  }

  if (!explicitScenarioNames_.empty()) {
    // Use explicit scenario assignment (e.g., from mpi-sppy hub wrapper)
    numSubdomains = (int)explicitScenarioNames_.size();

    // Build a stem→index map from the global list.
    // globalScenarioNames contains full paths (e.g., "/tmp/dir/Scenario0")
    // but explicit names from mpi-sppy are just stems ("Scenario0").
    std::unordered_map<std::string, int> stemToIndex;
    for (int i = 0; i < globalScenCount; ++i) {
      fs::path p(globalScenarioNames[i]);
      stemToIndex[p.filename().string()] = i;
    }

    localScenarioIndexList.resize(numSubdomains);
    for (int i = 0; i < numSubdomains; ++i) {
      // Try the name as-is first (stem), then strip any path
      auto it = stemToIndex.find(explicitScenarioNames_[i]);
      if (it == stemToIndex.end()) {
        fs::path ep(explicitScenarioNames_[i]);
        it = stemToIndex.find(ep.filename().string());
      }
      if (it == stemToIndex.end())
        throw std::runtime_error("Explicit scenario name '" + explicitScenarioNames_[i] +
                                 "' not found in problem directory " + problemName);
      localScenarioIndexList[i] = it->second;
    }
    // verbosity is not set until groupController::setup below, so read it here
    if (cfg.getOptionValue<int>("verbosity") >= 1)
      std::cerr << "globalScenCount=" << globalScenCount
                << " numSubdomains=" << numSubdomains
                << " (explicit assignment)" << std::endl;
  } else {
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

    scen.setup(cfg, globalScenarioNames[si]);
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

  // ---- DEBUG: dump local node metadata ----
  if (verbosity >= 1) {
    std::ostringstream os;
    os << "\n[R" << myRank << "] localNodes.size()=" << localNodes.size()
       << " globalNodeCount=" << globalNodeCount
       << " numSubdomains=" << numSubdomains
       << " globalScenCount=" << globalScenCount
       << "\n";

    for (size_t i = 0; i < localNodes.size(); ++i) {
      os << "[R" << myRank << "] i=" << i << " ";
      localNodes[i].pprint(os);
    }
    std::cerr << os.str() << std::flush;
  }
  // ----- end debug

  // It is a little uncool that we are requiring checkTree to be put in cfg elsewhere
  if (cfg.getOptionValue<bool>("check-tree")) {
      checkTreeConsistency();
  }

  setMaxConcurrentComms(globalNodeCount);  // in groupController
}

void spControllerBase::checkTreeConsistency()
{
  std::cerr << "\n[R" << myRank << "] === BEGIN TREE CONSISTENCY CHECK ===\n";

  for (auto& node : localNodes)
    {
      const std::string& nodeName = node.getName();
      int start = node.getDenseNonantStart();
      int len   = node.getDenseNonantLen();

      std::cerr << "[R" << myRank << "] Node "
                << nodeName
                << " serial=" << node.getSerialNumber()
                << " start=" << start
                << " len=" << len
                << " prob=" << node.getUncondProb()
                << "\n";

      auto& scenList = node.getNodeLocalScenarioIndexList();

      if (scenList.empty())
        {
          std::cerr << "[R" << myRank << "] WARNING: Node "
                    << nodeName << " has no scenarios on this rank.\n";
          continue;
        }

      // ===== PROBABILITY MASS CHECK =====
      double sumScenProb = 0.0;
      for (int s : scenList)
        sumScenProb += scenProb[s];

      double nodeProb = node.getUncondProb();

      std::cerr << "[R" << myRank
                << "] Node " << nodeName
                << " sumScenProb=" << std::setprecision(17) << sumScenProb
                << " nodeUncondProb=" << nodeProb
                << " diff=" << (sumScenProb - nodeProb)
                << "\n";

      // ===== 1. Check nonant count consistency =====
      int expectedLen = -1;

      for (int localScenI : scenList)
        {
          auto& scen = scenario(localScenI);
          int scenLen = scen.getNodeNonantCount(nodeName);

          if (expectedLen < 0)
            expectedLen = scenLen;
          else if (scenLen != expectedLen)
            {
              std::cerr << "\nFATAL: Nonant length mismatch at node "
                        << nodeName
                        << " scenario local=" << localScenI
                        << " expected=" << expectedLen
                        << " got=" << scenLen
                        << "\n";
              MPI_Abort(comm, 1);
            }
        }

      // ===== 2. Check nonant name ordering consistency =====
      std::vector<std::string> referenceNames;

      for (size_t k = 0; k < scenList.size(); ++k)
        {
          int localScenI = scenList[k];
          auto& scen = scenario(localScenI);

          const auto& nodeMap = scen.getTreeNodeDataMap();
          auto it = nodeMap.find(nodeName);

          if (it == nodeMap.end())
            {
              std::cerr << "\nFATAL: Scenario missing node "
                        << nodeName << "\n";
              MPI_Abort(comm,1);
            }

          const auto& names = it->second.nonAnts;

          if (k == 0)
            {
              referenceNames = names;
            }
          else
            {
              if (names.size() != referenceNames.size())
                {
                  std::cerr << "\nFATAL: Nonant name size mismatch at node "
                            << nodeName << "\n";
                  MPI_Abort(comm, 1);
                }

              for (size_t i = 0; i < names.size(); ++i)
                {
                  if (names[i] != referenceNames[i])
                    {
                      std::cerr << "\nFATAL: Nonant ORDER mismatch at node "
                                << nodeName
                                << "\n  Scenario local=" << localScenI
                                << "\n  index=" << i
                                << "\n  expected=" << referenceNames[i]
                                << "\n  got=" << names[i]
                                << "\n";
                      MPI_Abort(comm, 1);
                    }
                }
            }
        }

      // ===== 3. Check communicator size =====
      MPI_Comm nodeComm = node.getComm();
      if (nodeComm != MPI_COMM_NULL)
        {
          int commSize = -1;
          MPI_Comm_size(nodeComm, &commSize);

          std::cerr << "[R" << myRank
                    << "] Node " << nodeName
                    << " commSize=" << commSize
                    << " localScens=" << scenList.size()
                    << "\n";
        }

      // ===== 4. Check solver dense-nonant positions match (start,len) for ALL scenarios in node =====
      for (int localScenI : scenList)
        {
          auto& scen = scenario(localScenI);

          const auto& varIdxs = scen.getNodeVarIndexes(nodeName);
          if ((int)varIdxs.size() != len) {
            std::cerr << "\nFATAL: node " << nodeName
                      << " len=" << len
                      << " but solver node varIdx count=" << varIdxs.size()
                      << " (scenario local=" << localScenI << ")\n";
            MPI_Abort(comm, 1);
          }

          std::vector<int> nais;
          nais.reserve(varIdxs.size());
          for (int vidx : varIdxs) {
            int nai = scen.varIndexToNonantPos(vidx);
            nais.push_back(nai);
          }
          std::sort(nais.begin(), nais.end());

          bool ok = true;
          for (int j = 0; j < (int)nais.size(); ++j) {
            if (nais[j] != start + j) { ok = false; break; }
          }

          if (!ok) {
            std::cerr << "\nFATAL: node " << nodeName
                      << " dense segment mismatch for scenario local=" << localScenI << "!\n"
                      << "  expected contiguous [" << start << "," << (start+len-1) << "]\n"
                      << "  got nai positions (sorted): ";
            for (int v : nais) std::cerr << v << " ";
            std::cerr << "\n";
            MPI_Abort(comm, 1);
          }
        }
      std::cerr << "[R" << myRank << "] === END TREE CONSISTENCY CHECK ===\n\n";
    }
}

