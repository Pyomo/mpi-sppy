# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Utility functions for mmw, sequantial sampling and sample trees

import os
import numpy as np
import mpi4py.MPI as mpi
from sympy import factorint

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

def BFs_from_numscens(numscens,num_stages=2):
    #For now, we do not create unbalanced trees. 
    #If numscens cant be written as the product of a num_stages branching factors,
    #We take numscens <-numscens+1
    if num_stages == 2:
        return None
    else:
        for i in range(2**(num_stages-1)):
            n = numscens+i
            prime_fact = factorint(n)
            if sum(prime_fact.values())>=num_stages-1: #Checking that we have enough factors
                BFs = [0]*(num_stages-1)
                fact_list = [factor for (factor,mult) in prime_fact.items() for i in range(mult) ]
                for k in range(num_stages-1):
                    BFs[k] = np.prod([fact_list[(num_stages-1)*i+k] for i in range(1+len(fact_list)//(num_stages-1)) if (num_stages-1)*i+k<len(fact_list)])
                return BFs
        raise RuntimeError("BFs_from_numscens is not working correctly. Did you take num_stages>=2 ?")
        
def is_sorted(nodelist):
    #Take a list of scenario_tree.ScenarioNode and check that it is well constructed
    parent=None
    for (t,node) in enumerate(nodelist):
        if (t+1 != node.stage) or (node.parent_name != parent):
            raise RuntimeError("The node list is not well-constructed"
                               f"The stage {node.stage} node is the {t+1}th element of the list."
                               f"The node {node.name} has a parent named {node.parent_name}, but is right after the node {parent}")
        parent = node.name
        
def writetxt_xhat(xhat,path="xhat.txt",num_stages=2):
    if num_stages ==2:
        np.savetxt(path,xhat['ROOT'])
    else:
        raise RuntimeError("Only 2-stage is suported to write/read xhat to a file")

def readtxt_xhat(path="xhat.txt",num_stages=2,delete_file=False):
    if num_stages==2:
        xhat = {'ROOT': np.loadtxt(path)}
    else:
        raise RuntimeError("Only 2-stage is suported to write/read xhat to a file")
    if delete_file and global_rank ==0:
        os.remove(path)        
    return(xhat)

def write_xhat(xhat,path="xhat.npy",num_stages=2):
    if num_stages==2:
        np.save(path,xhat['ROOT'])
    else:
        raise RuntimeError("Only 2-stage is suported to write/read xhat to a file")
    

def read_xhat(path="xhat.npy",num_stages=2,delete_file=False):
    if num_stages==2:
        xhat = {'ROOT': np.load(path)}
    else:
        raise RuntimeError("Only 2-stage is suported to write/read xhat to a file")
    if delete_file and global_rank ==0:
        os.remove(path)
    return(xhat)