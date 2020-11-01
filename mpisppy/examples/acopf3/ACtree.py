# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
#Tree ideas; dlw Fall 2019
# Stage numbers and scenario numbers are one-based, but lists are all zero-based
# Tree node numbers used in node names are zero-based.
# NOTE: we *do* have leaf nodes, but mpisppy just completes the scen with them
# NOTE: mpisppy needs non-leaf, non-ROOT nodes to have names ending in a serial #
# CHANGE: Jan 2020: fail prob is per line

import numpy as np
import copy

def FixFast(minutes):
    return True

"""
At each scenario tree node:
   a link can fail and/or
   a previously failed link can be repaired; prob(repair) is IFR
   at the root and at any node that has nothing broken,
      the child None is always present and has high probability

"""

class ACTree():
    """
    Data for a scenario tree for ACOPF.

    Args:
        NumStages (int >= 2) Number of stages
        BFs (list of NumStages ints) Branching factor at each stage
        seed (int) the psuedo-random number seed
        acstream (np.random.RandomState) the random number stream to use
        FailProb (float) prob for each line to fail
        StageDurations (list of Numstages ints) number of minutes per stage
        Repairer (function with argument t in minutes) returns True for repair
        LineList (list of int) All lines in the electric grid
    """
    def __init__(self, NumStages, BFs, seed, acstream, FailProb,
                 StageDurations, Repairer, LineList):
        self.NumStages = NumStages
        self.BFs = BFs
        self.seed = seed
        acstream.seed(seed)
        self.FailProb = FailProb
        self.StageDurations = StageDurations
        self.Repairer = Repairer
        self.LineList = LineList

        self.numscens = 1
        for s in range(self.NumStages-1):
            self.numscens *= self.BFs[s]
        
        # now make the tree by making the root
        self.rootnode = TreeNode(None, self,
                                 [i+1 for i in range(self.numscens)],
                                 "ROOT", 1.0, acstream)

    def Nodes_for_Scenario(self, scen_num):
        """
        Return a list of nodes for a given scenario number (one-based)
        """
        assert(scen_num <= self.numscens)
        retlist = [self.rootnode]
        # There must be a clever arithmetic way to do this, but...
        for stage in range(2, self.NumStages+1):
            for kid in retlist[-1].kids:
                if scen_num in kid.ScenarioList:
                    retlist.append(kid)
                    break
        assert(len(retlist) == self.NumStages)
        return retlist

    def All_Nonleaf_Nodenames(self):
        """ Return a list of all non-leaf node names"""
        # there is a arithmetic way, but I will use the general tree way
        def _progenynames(node):
            # return my name and progeny names if they are not leaves
            retval = [node.Name]
            if node.stage < self.NumStages - 1: # avoid leaves
                for kid in node.kids:
                    retval += _progenynames(kid)
            return retval
            
        allnonleaf = _progenynames(self.rootnode)
        return allnonleaf
        
            
class TreeNode():
    """
    Data for a tree node, but the node creates its own children

    Args:
        Parent (TreeNode) parent node
        TreeInfo (Tree) the tree to which I belong
        ScenarioList (list of int) computed by caller
        Name (str) for output
        CondProb (float) conditional probability

    Attributes:
        FailedLines (list of (line, minutesout)) the lines currently out
        LinesUp (list of int) the lines that are operational
        kids (list of nodes) the children of this node

    """
    def __init__(self, Parent, TreeInfo, ScenarioList, Name, CondProb, acstream):

        self.sn = 1 # to attach serial numbers where needed
        self.CondProb = CondProb
        self.Name = Name
        self.Parent = Parent
        self.ScenarioList = ScenarioList
        if Parent is None:
            # Root node
            self.stage = 1
            self.FailedLines = []
            self.LinesUp = copy.deepcopy(TreeInfo.LineList)
        else:
            self.stage = Parent.stage+1
            self.FailedLines = copy.deepcopy(Parent.FailedLines)
            self.LinesUp = copy.deepcopy(Parent.LinesUp)
            # bring lines up? (mo is minutes out)
            removals = list()
            for ell in range(len(self.FailedLines)):
                line, mo = self.FailedLines[ell]
                if TreeInfo.Repairer(mo):
                    removals.append((line, mo))
                    self.LinesUp.append(line)
                else:
                    mo += TreeInfo.StageDurations[stage-1]
                    self.FailedLines[ell] = (line, mo)
            for r in removals:
                self.FailedLines.remove(r)

            # bring lines down?
            for line in self.LinesUp:
                if acstream.rand() < TreeInfo.FailProb:
                    self.LinesUp.remove(line)
                    self.FailedLines.append\
                        ((line, TreeInfo.StageDurations[self.stage-1]))
        if self.stage <= TreeInfo.NumStages:
            # spawn children
            self.kids = []
            if self.stage < TreeInfo.NumStages:
                bf = TreeInfo.BFs[self.stage-1]
                snstr = "_sn"+str(self.sn)
                self.sn += 1 # serial number for non-leaf, non-ROOT nodes
            else:
                bf = 1  # leaf node
                snstr = ""
            for b in range(bf):
                # divide up the scenario list
                plist = self.ScenarioList # typing aid
                first = b*len(plist) // bf 
                last = (b+1)*len(plist) // bf
                scenlist = plist[first: last]
                newname = self.Name + "_" + str(b)
                if self.stage < TreeInfo.NumStages:
                    prevbf = TreeInfo.BFs[self.stage-2]
                    self.kids.append(TreeNode(self, TreeInfo, scenlist,
                                              newname, 1/prevbf, acstream))
                    
    def pprint(self):
        print("Node Name={}, Stage={}".format(self.Name, self.stage))
        if self.Parent is not None:
            print("   Parent Name={}".format(self.Parent.Name))
        else:
            print("   (no parent)")
        print("   FailedLines={}".format(self.FailedLines))

if __name__ == "__main__":

    acstream = np.random.RandomState()
    
    testtree = ACTree(3, [2, 2], 1134, acstream,
                      0.2, [5,15,30], FixFast, [0,1,2,3,4,5])
    for sn in range(1,testtree.numscens+1):
        print ("nodes in scenario {}".format(sn))
        for node in testtree.Nodes_for_Scenario(sn):
            # print ("  ",node.Name)
            # print("      ", node.FailedLines)
            # print("      ", node.ScenarioList)
            node.pprint()
