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
import mpisppy.utils.xhat_eval as xhat_eval
import mpisppy.utils.amalgomator as ama
import mpisppy.confidence_intervals.sample_tree as sample_tree

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

def scalable_BFs(numscens, ref_BFs):
    '''
    This utilitary find a good branching factor list to create a scenario tree
    containing at least numscens leaf nodes, and scaled like ref_BFs.
    For instance, if numscens=233 and reef_BFs=[5,3,2], it returns [10,6,4], 
    branching factors for a 240-leafs tree
    
    NOTE: This method increasing in priority first stages branching factors,
          so that a branching factor is an increasing function of numscens

    Parameters
    ----------
    numscens : int
        Number of leaf nodes/scenarios of the tree.
    ref_BFs : list of int
        Reference shape of the branching factors. It length must be equal to
        number_of_stages-1

    Returns
    -------
    new_BFs
        DESCRIPTION.

    '''
    numstages = len(ref_BFs)+1
    if numscens < 2**(numstages-1):
        return [2]*(numstages-1)
    mult_coef = (numscens/np.prod(ref_BFs))**(1/(numstages-1))
    new_BFs = np.maximum(np.floor(np.array(ref_BFs)*mult_coef),1.) #BFs have to be positive integers
    i=0
    while np.prod(new_BFs)<numscens:
        if i == numstages-1:
            raise RuntimeError("scalable BFs is failing")
        new_BFs[i]+=1
        i+=1
    new_BFs = list(new_BFs.astype(int))
    return new_BFs
        
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

def correcting_numeric(G,relative_error=True,threshold=1e-4,objfct=None):
    #Correcting small negative of G due to numerical error while solving EF 
    if relative_error:
        if objfct is None:
            raise RuntimeError("We need a value of the objective function to remove numerically negative G")
        elif (G<= -threshold*np.abs(objfct)):
            raise RuntimeWarning(f"WARNING: The gap estimator is anormaly negative : {G}")
            return G
        else:
            return max(0,G)
    else:
        if (G<=-threshold):
            raise RuntimeWarning(f"WARNING: The gap estimator is anormaly negative : {G}")
            return G
        else: 
            return max(0,G)           

def gap_estimators(xhat_one,
                   mname, 
                   solving_type="EF-2stage",
                   scenario_names=None,
                   sample_options=None,
                   ArRP=1,
                   scenario_creator_kwargs={}, 
                   scenario_denouement=None,
                   solvername='gurobi', 
                   solver_options=None,
                   verbose=True,
                   objective_gap=False
                   ):
    ''' Given a xhat, scenario names, a scenario creator and options, 
    gap_estimators creates a scenario tree and the associatd estimators 
    G and s from §2 of [bm2011].
    Returns G and s evaluated at xhat.
    If ArRP>1, G and s are pooled, from a number ArRP of estimators,
        computed with different scenario trees.
    

    Parameters
    ----------
    xhat_one : dict
        A candidate first stage solution
    mname: str
        Name of the reference model, e.g. 'mpisppy.tests.examples.farmer'.
    solving_type: str, optional
        The way we solve the approximate problem. Can be "EF-2stage" (default)
        or "EF-mstage".
    scenario_names: list, optional
        List of scenario names used to compute G_n and s_n. Default is None
        Must be specified for 2 stage, but can be missing for multistage
    sample_options: dict, optional
        Only for multistage. Must contain a 'seed' and a 'BFs' attribute,
        specifying the starting seed and the branching factors 
        of the scenario tree
    ArRP:int,optional
        Number of batches (we create a ArRP model). Default is 1 (one batch).
    scenario_creator_kwargs: dict, optional
        Additional arguments for scenario_creator. Default is {}
    scenario_denouement: function, optional
        Function to run after scenario creation. Default is None.
    solvername : str, optional
        Solver. Default is 'gurobi'
    solver_options: dict, optional
        Solving options. Default is None
    verbose: bool, optional
        Should it print the gap estimator ? Default is True
    objective_gap: bool, optional
        Returns a gap estimate around approximate objective value

    BFs: list, optional
        Only for multistage. List of branching factors of the sample scenario tree.

    Returns
    -------
    G_k and s_k, gap estimator and associated standard deviation estimator.

    '''
    if solving_type not in ["EF-2stage","EF-mstage"]:
        raise RuntimeError("Only EF solve for the approximate problem is supported yet.")
    else:
        is_multi = (solving_type=="EF-mstage")
    
    
        
    if is_multi:
        try:
            BFs = sample_options['BFs']
            start = sample_options['seed']
        except (TypeError,KeyError,RuntimeError):
            raise RuntimeError('For multistage problems, sample_options must be a dict with BFs and seed attributes.')
    else:
        start = sputils.extract_num(scenario_names[0])
    if ArRP>1: #Special case : ArRP, G and s are pooled from r>1 estimators.
        if is_multi:
            raise RuntimeError("Pooled estimators are not supported for multistage problems yet.")
        n = len(scenario_names)
        if(n%ArRP != 0):
            raise RuntimeWarning("You put as an input a number of scenarios"+\
                                 f" which is not a mutliple of {ArRP}.")
            n = n- n%ArRP
        G =[]
        s = []

        for k in range(ArRP):
            scennames = scenario_names[k*(n//ArRP):(k+1)*(n//ArRP)]
            tmp = gap_estimators(xhat_one, mname,
                                   solvername=solvername,
                                   scenario_names=scennames, ArRP=1,
                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                   scenario_denouement=scenario_denouement,
                                   solver_options=solver_options,
                                   solving_type=solving_type
                                   )
            G.append(tmp['G'])
            s.append(tmp['s']) 

        #Pooling
        G = np.mean(G)
        s = np.linalg.norm(s)/np.sqrt(n//ArRP)
        return {"G": G, "s": s, "seed": start}
    

    #A1RP
    
    #We start by computing the optimal solution to the approximate problem induced by our scenarios
    if is_multi:
        #Sample a scenario tree: this is a subtree, but starting from stage 1
        samp_tree = sample_tree.SampleSubtree(mname,
                                              xhats =[],
                                              root_scen=None,
                                              starting_stage=1, 
                                              BFs=BFs,
                                              seed=start, 
                                              options=scenario_creator_kwargs,
                                              solvername=solvername,
                                              solver_options=solver_options)
        samp_tree.run()
        start += sputils.number_of_nodes(BFs)
        ama_object = samp_tree.ama
    else:
        #We use amalgomator to do it

        ama_options = dict(scenario_creator_kwargs)
        ama_options['start'] = start
        ama_options['num_scens'] = len(scenario_names)
        ama_options['EF_solver_name'] = solvername
        ama_options['EF_solver_options'] = solver_options
        ama_options[solving_type] = True
        ama_object = ama.from_module(mname, ama_options,use_command_line=False)
        ama_object.scenario_names = scenario_names
        ama_object.verbose = False
        ama_object.run()
        start += len(scenario_names)
        
    #Optimal solution of the approximate problem
    zstar = ama_object.best_outer_bound
    #Associated policies
    xstars = sputils.nonant_cache_from_ef(ama_object.ef)
    
    #Then, we evaluate the fonction value induced by the scenario at xstar.
    
    if is_multi:
        # Find feasible policies (i.e. xhats) for every non-leaf nodes
        if len(samp_tree.ef._ef_scenario_names)>1:
            local_scenarios = {sname:getattr(samp_tree.ef,sname) for sname in samp_tree.ef._ef_scenario_names}
        else:
            local_scenarios = {samp_tree.ef._ef_scenario_names[0]:samp_tree.ef}
        xhats,start = sample_tree.walking_tree_xhats(mname,
                                                    local_scenarios,
                                                    xhat_one['ROOT'],
                                                    BFs,
                                                    start,
                                                    scenario_creator_kwargs,
                                                    solvername=solvername,
                                                    solver_options=solver_options)
        
        #Compute then the average function value with this policy
        scenario_creator_kwargs = samp_tree.ama.kwargs
        all_nodenames = sputils.create_nodenames_from_BFs(BFs)
    else:
        #In a 2 stage problem, the only non-leaf is the ROOT node
        xhats = xhat_one
        all_nodenames = None
    
    xhat_eval_options = {"iter0_solver_options": None,
                         "iterk_solver_options": None,
                         "display_timing": False,
                         "solvername": solvername,
                         "verbose": False,
                         "solver_options":solver_options}
    ev = xhat_eval.Xhat_Eval(xhat_eval_options,
                            scenario_names,
                            ama_object.scenario_creator,
                            scenario_denouement,
                            scenario_creator_kwargs=scenario_creator_kwargs,
                            all_nodenames = all_nodenames)
    #Evaluating xhat and xstar and getting the value of the objective function 
    #for every (local) scenario
    ev.evaluate(xhats)
    objs_at_xhat = ev.objs_dict
    ev.evaluate(xstars)
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
    if global_rank==0 and verbose:
        print(f"G = {G}")
    sample_var = (ssq - G**2)/(1-prob_sqnorm) #Unbiased sample variance
    s = np.sqrt(sample_var)
    
    use_relative_error = (np.abs(zstar)>1)
    G = correcting_numeric(G,objfct=obj_at_xhat,
                           relative_error=use_relative_error)
    if objective_gap:
        if is_multi:
            return {"G":G,"s":s,"zhats": [obj_at_xhat], "seed":start} 
        else:
            return {"G":G,"s":s,"zhats": eval_scen_at_xhat, "seed":start} 
    else:
        return {"G":G,"s":s,"seed":start}