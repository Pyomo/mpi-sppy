 #pragma once

// The class for a scenario tree node

#include "utils/utilities.hpp"
#include "scenario.hpp"
#include "solver_interfaces/TreeNodeData.hpp"

class ScenarioTreeNode {
public:
  ScenarioTreeNode() = default;
  ~ScenarioTreeNode() { if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm); }

  ScenarioTreeNode(const ScenarioTreeNode&) = delete;
  ScenarioTreeNode& operator=(const ScenarioTreeNode&) = delete;

  ScenarioTreeNode(ScenarioTreeNode&& other) noexcept {
    *this = std::move(other);
  }

  ScenarioTreeNode& operator=(ScenarioTreeNode&& other) noexcept {
    if (this == &other) return *this;

    // free what we currently own
    if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);

    // move plain fields
    nodeName = std::move(other.nodeName);
    nodeData = other.nodeData;
    uncondProb = other.uncondProb;
    nodeLocalScenarioIndexList = std::move(other.nodeLocalScenarioIndexList);
    denseNonantStart = other.denseNonantStart;
    denseNonantLen = other.denseNonantLen;
    serialNumber = other.serialNumber;
    reduceInBuf = std::move(other.reduceInBuf);
    reduceOutBuf = std::move(other.reduceOutBuf);

    // transfer ownership of communicator handle
    comm = other.comm;
    other.comm = MPI_COMM_NULL;  // CRITICAL: prevent double-free

    return *this;
  }
  const std::string& getName() const { return nodeName; }
  int getSerialNumber() const { return serialNumber; }

  // Note that setup does not assign the comm (which is done by spController's setup)
  // IMPORTANT: we assume setup has been called for the parent
  void setup(const std::string &Name,
             const TreeNodeData& ndData,
             Scenario& aScenario,
             ScenarioTreeNode* parent=nullptr) {
    nodeName = Name;
    nodeData = ndData;
    serialNumber = ndData.serialNumber;
    if (parent == nullptr) {
      assert(nodeName == "ROOT"  && "parent is nullptr");
      uncondProb = 1.0;
      denseNonantStart = 0;
    } else {
      uncondProb = parent->getUncondProb() * ndData.condProb;
      denseNonantStart = parent->getDenseNonantStart() + parent->getDenseNonantLen();
    }
    denseNonantLen = aScenario.getNodeNonantCount(nodeName);
    initReductionBuffers();
  }
  
  void setComm(MPI_Comm new_comm) {
    // std::cerr << "debug in treeNode.hpp setting comm" << std::endl;
    if (new_comm == comm) return;
    if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
    comm = new_comm;          // now owned; will be freed in ~ScenarioTreeNode
    if (!ensure_comm_ok("setComm in treeNode.hpp", comm)) {
      
    }
  }

  MPI_Comm& getComm() {return comm;}
  
  void pprint(std::ostream& out) const;

  int getDenseNonantStart() const {
    return denseNonantStart;
  }
  int getDenseNonantLen() const {
    return denseNonantLen;
  }
  double getUncondProb() const {
    return uncondProb; // node probability
  }
  
  std::vector<double>& inBuf()  { return reduceInBuf; }
  std::vector<double>& outBuf() { return reduceOutBuf; }
  const std::vector<double>& outBuf() const { return reduceOutBuf; }

  // For now, we will trust that there is only one caller that avoids duplicates.
  void addLocalScenIndex(int localScenI) {
    nodeLocalScenarioIndexList.push_back(localScenI);
  }

  std::vector<int>& getNodeLocalScenarioIndexList() {
    return nodeLocalScenarioIndexList;
  }
  const std::vector<int>& getNodeLocalScenarioIndexList() const {
    return nodeLocalScenarioIndexList;
  }  
protected:
  std::string nodeName;

  // (scenarios also have the tree node data from the disk, because they get it from below)
  TreeNodeData nodeData;
  
  double uncondProb = 0;

  // These indexes are into the list for this rank (on spControllerBase).
  std::vector<int> nodeLocalScenarioIndexList;

  int denseNonantStart = 0;
  int denseNonantLen = 0;
  MPI_Comm comm = MPI_COMM_NULL;
  int serialNumber;   // position in tree counting top to bottom, left to right (comm color)

  std::vector<double> reduceInBuf;
  std::vector<double> reduceOutBuf;
  // this needs to be called after denseNonantLen is known; i.e., after setup
  void initReductionBuffers() {
    reduceInBuf.assign(2 * denseNonantLen, 0.0);
    reduceOutBuf.assign(2 * denseNonantLen, 0.0);
  }

};

inline void ScenarioTreeNode::pprint(std::ostream& out) const {
  int commSz = -1, commRk = -1;
  char commName[MPI_MAX_OBJECT_NAME];
  int nameLen = 0;

  if (comm != MPI_COMM_NULL) {
    MPI_Comm_size(comm, &commSz);
    MPI_Comm_rank(comm, &commRk);
    MPI_Comm_get_name(comm, commName, &nameLen);
  } else {
    std::snprintf(commName, sizeof(commName), "MPI_COMM_NULL");
    nameLen = (int)std::strlen(commName);
  }

  out << "Node{name=" << nodeName
      << " serial=" << serialNumber
      << " uncondProb=" << uncondProb
      << " naStart=" << denseNonantStart
      << " xLen=" << denseNonantLen
      << " localScens=" << nodeLocalScenarioIndexList.size()
      << " commName=" << std::string(commName, nameLen)
      << " commRank=" << commRk
      << " commSize=" << commSz
      << "}\n";
}


//==== local utility =====
inline std::string strip_suffix(const std::string& input) {
  // returns the parent name if using the mpisppy node naming convention  
  size_t pos = input.rfind('_');
  if (pos == std::string::npos) {
    throw std::runtime_error("No suffix for non-ROOT node name=" + input);
  }
  return input.substr(0, pos);
}
