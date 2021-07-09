# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Utility functions for mmw, sequantial sampling and sample trees

import os
import numpy as np
import mpi4py.MPI as mpi
from sympy import factorint
import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
from mpisppy import global_toc
import mpisppy.utils.xhat_eval

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

def gap_estimators(xhat,solvername, scenario_names, scenario_creator, ArRP=1,
                   scenario_creator_kwargs={}, scenario_denouement=None,
                   solver_options=None,solving_type="EF-2stage"):
    ''' Given a xhat, scenario names, a scenario creator and options, create
    the scenarios and the associatd estimators G and s from §2 of [bm2011].
    Returns G and s evaluated at xhat.
    If ArRP>1, G and s are pooled, from a number ArRP of estimators,
        computed on different batches.
    

    Parameters
    ----------
    xhat : xhat object
        A candidate solution
    solvername : str
        Solver
    scenario_names: list
        List of scenario names used to compute G_n and s_n.
    scenario_creator: function
        A method creating scenarios.
    ArRP:int,optional
        Number of batches (we create a ArRP model). Default is 1 (no batches).
    scenario_creator_kwargs: dict, optional
        Additional arguments for scenario_creator. Default is {}
    scenario_denouement: function, optional
        Function to run after scenario creation. Default is None.
    solver_options: dict, optional
        Solving options. Default is None
    solving_type: str, optionale
        The way we solve the approximate problem. Can be "EF-2stage" (default)
        or "EF-mstage".

    Returns
    -------
    G_k and s_k, gap estimator and associated standard deviation estimator.

    '''
    if solving_type not in ["EF-2stage","EF-mstage"]:
        raise RuntimeError("Only EF solve for the approximate problem is supported yet.")
    
    if ArRP>1: #Special case : ArRP, G and s are pooled from r>1 estimators.
        n = len(scenario_names)
        if(n%ArRP != 0):
            raise RuntimeWarning("You put as an input a number of scenarios"+\
                                 f" which is not a mutliple of {ArRP}.")
            n = n- n%ArRP
        G =[]
        s = []
        for k in range(n//ArRP):
            scennames = scenario_names[k*ArRP,(k+1)*ArRP -1]
            tmpG,tmps = gap_estimators(xhat, solvername, scennames, 
                                       scenario_creator, ArRP=1,
                                       scenario_creator_kwargs=scenario_creator_kwargs,
                                       scenario_denouement=scenario_denouement,
                                       solver_options=solver_options,
                                       solving_type=solving_type)
            G.append(tmpG)
            s.append(tmps)
        #Pooling
        G = np.mean(G)
        s = np.linalg.norm(s)/np.sqrt(n//ArRP)
        return(G,s)
    
    #A1RP
    #We start by computing the solution to the approximate problem induced by our scenarios
    ef = sputils.create_EF(
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs=scenario_creator_kwargs,
        suppress_warnings=True,
        )
    
    solver = pyo.SolverFactory(solvername)
    if 'persistent' in solvername:
        solver.set_instance(ef, symbolic_solver_labels=True)
        solver.solve(tee=False)
    else:
        solver.solve(ef, tee=False, symbolic_solver_labels=True,)

    xstar = sputils.nonant_cache_from_ef(ef)
    options = {"iter0_solver_options": None,
             "iterk_solver_options": None,
             "display_timing": False,
             "solvername": solvername,
             "verbose": False,
             "solver_options":solver_options}
    

    scenario_creator_kwargs['num_scens'] = len(scenario_names)
    ev = xhat_eval.Xhat_Eval(options,
                    scenario_names,
                    scenario_creator,
                    scenario_denouement,
                    scenario_creator_kwargs=scenario_creator_kwargs)
    #Evaluating xhat and xstar and getting the value of the objective function 
    #for every (local) scenario
    global_toc(f"xhat={xhat}")
    global_toc(f"xstar={xstar}")
    ev.evaluate(xhat)
    objs_at_xhat = ev.objs_dict
    ev.evaluate(xstar)
    objs_at_xstar = ev.objs_dict
    
    eval_scen_at_xhat = []
    eval_scen_at_xstar = []
    scen_probs = []
    for k,s in ev.local_scenarios.items():
        eval_scen_at_xhat.append(objs_at_xhat[k])
        eval_scen_at_xstar.append(objs_at_xstar[k])
        scen_probs.append(s._mpisppy_probability)

    
    scen_gaps = np.array(eval_scen_at_xhat)-np.array(eval_scen_at_xstar)
    local_gap = np.dot(scen_gaps,scen_probs)
    local_ssq = np.dot(scen_gaps**2,scen_probs)
    local_prob_sqnorm = np.linalg.norm(scen_probs)**2
    local_obj_at_xhat = np.dot(eval_scen_at_xhat,scen_probs)
    local_estim = np.array([local_gap,local_ssq,local_prob_sqnorm,local_obj_at_xhat])
    global_estim = np.zeros(4)
    ev.mpicomm.Allreduce(local_estim, global_estim, op=mpi.SUM) 
    G,ssq, prob_sqnorm,obj_at_xhat = global_estim
    if global_rank==0:
        print(f"G = {G}")
    sample_var = (ssq - G**2)/(1-prob_sqnorm) #Unbiased sample variance
    s = np.sqrt(sample_var)
    G = MMWci.correcting_numeric(G,objfct=obj_at_xhat)
    return [G,s]