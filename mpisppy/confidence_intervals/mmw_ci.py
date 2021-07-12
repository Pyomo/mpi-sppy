# Code to evaluate a given x-hat given as a nonant-cache, and the MMW confidence interval.
# To test: python mmw_ci.py --num-scens=3  --MMW-num-batches=3 --MMW-batch-size=3


import mpi4py.MPI as mpi
import argparse
import numpy as np
import scipy.stats
import importlib
import os
from mpisppy import global_toc
    
fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

import mpisppy.utils.amalgomator as ama
import mpisppy.utils.xhat_eval as xhat_eval
import mpisppy.utils.sputils as sputils
import mpisppy.confidence_intervals.sample_tree as sample_tree
import mpisppy.confidence_intervals.ciutils as ciutils

def remove_None(d):
    if d is None:
        return {}
    d_copy = {}
    for (key,value) in d.items():
        if value is not None:
            d_copy[key] = value
    return d_copy



class MMWConfidenceIntervals():
    """Takes a model and options as input. 

    Args:
        refmodel (str): path of the model we use (e.g. farmer, uc)
        options (dict): useful options to run amalgomator or xhat_eval, 
                        including EF_solver_options and EF_solver_name
                        May include the options used to compute xhat
        xhat_one (dict): Non-anticipative first stage solution, computed before
        num_batches (int): Number of batches used to compute the MMW estimator
        batch_size (int): Size of MMW batches, default None. 
                If batch_size is None, then batch_size=options['num_scens'] is used
        start (int): first scenario used to run the MMW estimator, 
                    default None
                If start is None, then start=options[num_scens]+options[start] is used
    
    Note:
        Options can include the following things:
            -The type of our solving. for now, MMWci only support EF, so it must
            have an attribute 'EF-2stage' or 'EF-mstage' set equal to True
            - Solver-related options ('EF_solver_name' and 'EF_solver_options')
            
    
    """
    def __init__(self,
                 refmodel,
                 options,
                 xhat_one,
                 num_batches,
                 batch_size=None,
                 start=None,
                 verbose=True,
                 ):
        self.refmodel = importlib.import_module(refmodel)
        self.refmodelname = refmodel
        self.options = options
        self.xhat_one = xhat_one
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.num_scens_xhat = options["num_scens"] if ("num_scens" in options) else 0
        
        #Getting the start
        if start is not None :
            self.start = start
        ScenCount = self.num_scens_xhat + (
            self.options['start'] if ("start" in options) else 0)
        if start is None :
            self.start = ScenCount
        elif start < ScenCount :
            raise RuntimeWarning(
                "Scenarios used to compute xhat_one may be used in MMW")
            
        #Type of our problem
        if ama._bool_option(options, "EF-2stage"):
            self.type = "EF-2stage"
            self.multistage = False
            self.numstages = 2
        elif ama._bool_option(options, "EF-mstage"):
            self.type = "EF-mstage"
            self.multistage = True
            self.numstages = len(options['BFs'])+1
        else:
            raise RuntimeError(
                "Only EF is supported. options should get an attribute 'EF-2stage' or 'EF-mstage' set to True")
        
        #Check if refmodel and args have all needed attributes
        everything = ["scenario_names_creator",
                 "scenario_creator",
                 "kw_creator"]  # denouement can be missing.
        if self.multistage:
            everything[0] = "sample_tree_scen_creator"
    
        you_can_have_it_all = True
        for ething in everything:
            if not hasattr(self.refmodel, ething):
                print(f"Module {refmodel} is missing {ething}")
                you_can_have_it_all = False
        if not you_can_have_it_all:
            raise RuntimeError(f"Module {refmodel} not complete for MMW")
        
        you_can_have_it_all = True
        for ething in ["num_scens","EF_solver_name"]:
            if not ething in self.options:
                print(f"Argument list is missing {ething}")
                you_can_have_it_all = False
        if not you_can_have_it_all:
            raise RuntimeError("Argument list not complete for MMW")   
            
            
    def run(self,confidence_level=0.95):
        # We get the MMW right term, then xhat, then the MMW left term.


        #Compute the nonant xhat (the one used in the left term of MMW (9) ) using
        #                                                        the first scenarios
        
        ############### get the parameters
        start = self.start
        scenario_denouement = self.refmodel.scenario_denouement

        #Introducing batches otpions
        num_batches = self.num_batches
        bs=self.batch_size
        batch_size = bs if (bs is not None) else start #is None : take size_batch=num_scens        
        sample_options = self.options
        
        #Some options are specific to 2-stage or multi-stage problems
        if self.multistage:
            sampling_BFs = ciutils.BFs_from_numscens(batch_size,self.numstages)
            #TODO: Change this to get a more logical way to compute BFs
            batch_size = np.prod(sampling_BFs)
        else:
            sampling_BFs = None
            
        sample_options['num_scens'] = batch_size
        sample_options['_mpisppy_probability'] = 1/batch_size
        scenario_creator_kwargs=self.refmodel.kw_creator(sample_options)
        sample_scen_creator = self.refmodel.scenario_creator
        
        #Solver settings
        solvername = self.options['EF_solver_name']
        solver_options = self.options['EF_solver_options'] if 'EF_solver_options' in self.options else None
        solver_options = remove_None(solver_options)
            
        #Now we compute for each batch the whole Gn term from MMW (9)

        G = np.zeros(num_batches) #the Gbar of MMW (10)
        #we will compute the mean via a loop (to be parallelized ?)
        
        
        for i in range(num_batches) :
            scenario_names = self.refmodel.scenario_names_creator(batch_size,start=start)
            estim = ciutils.gap_estimators(self.xhat_one,solvername,
                                            scenario_names, self.refmodelname, 
                                            ArRP=1,
                                            scenario_creator_kwargs=scenario_creator_kwargs,
                                            scenario_denouement=scenario_denouement,
                                            solver_options=solver_options,
                                            solving_type=self.type,
                                            BFs=sampling_BFs)
            Gn = estim['G']
            start = estim['seed']
            # if self.multistage:
            #     #Sample a scenario tree: this is a subtree, but starting from stage 1
            #     samp_tree = sample_tree.SampleSubtree(self.refmodelname,
            #                                           xhats =[],
            #                                           root_scen=None,
            #                                           starting_stage=1, 
            #                                           BFs=sampling_BFs, 
            #                                           seed=start, 
            #                                           options=sample_options,
            #                                           solvername=solvername,
            #                                           solver_options=solver_options)
            #     samp_tree.run()
            #     start += sputils.number_of_nodes(sampling_BFs)
            #     ama_object = samp_tree.ama
            # else:
            #     #First we compute the right term of MMW (9)
            #     MMW_scenario_names = self.refmodel.scenario_names_creator(
            #         batch_size, start=start)
                
                
            #     #We use amalgomator to do it
    
            #     ama_options = dict(scenario_creator_kwargs)
            #     ama_options['start'] = start
            #     ama_options['EF_solver_name'] = solvername
            #     ama_options['EF_solver_options'] = solver_options
            #     ama_options[self.type] = True
            #     ama_object = ama.from_module(self.refmodelname, ama_options,use_command_line=False)
            #     ama_object.verbose = self.verbose
            #     ama_object.run()
            #     start += batch_size
                
            # #Find the right term of MMW
            # MMW_right_term = ama_object.best_outer_bound
                
            # if self.multistage:
            #     #Now find feasible policies (i.e. xhats) for every non-leaf nodes
            #     MMW_scenario_names = samp_tree.ef._ef_scenario_names
            #     local_scenarios = {sname:getattr(samp_tree.ef,sname) for sname in MMW_scenario_names}
            #     xhats,start = sample_tree.walking_tree_xhats(self.refmodelname,
            #                                                 local_scenarios,
            #                                                 self.xhat_one,
            #                                                 sampling_BFs,
            #                                                 start,
            #                                                 sample_options,
            #                                                 solvername=solvername,
            #                                                 solver_options=solver_options)
                
            #     #Compute then the average function value with this policy
            #     sample_scen_creator = samp_tree.sample_creator
            #     scenario_creator_kwargs = samp_tree.ama.kwargs
            #     all_nodenames = sputils.create_nodenames_from_BFs(sampling_BFs)
            # else:
            #     #In a 2-stage problem, we do not need to find feasible policies apart from xhat_one
            #     xhats = self.xhat_one
            #     all_nodenames = None
                
            # xhat_eval_options = {"iter0_solver_options": None,
            #                       "iterk_solver_options": None,
            #                       "display_timing": False,
            #                       "solvername": solvername,
            #                       "verbose": False,
            #                       "solver_options":solver_options}
            # ev = xhat_eval.Xhat_Eval(xhat_eval_options,
            #                         MMW_scenario_names,
            #                         sample_scen_creator,
            #                         scenario_denouement,
            #                         scenario_creator_kwargs=scenario_creator_kwargs,
            #                         all_nodenames = all_nodenames)
            # MMW_left_term = ev.evaluate(xhats)
                            
    
            # #Now we can compute MMW (9)
            # Gn = MMW_left_term-MMW_right_term
            # use_relative_error = (np.abs(MMW_right_term)>1)
            # Gn = ciutils.correcting_numeric(Gn,
            #                         relative_error=use_relative_error,
            #                         objfct=MMW_right_term)
            
            if(self.verbose):
                global_toc(f"Gn={Gn} for the batch {i}")  # Left term of LHS of (9)
            G[i]=Gn             
        s = np.std(G) #Standard deviation
        Gbar = np.mean(G)
        t = scipy.stats.t.ppf(confidence_level,num_batches-1)
        epsilon = t*s/np.sqrt(num_batches)
        gap_inner_bound =  Gbar+epsilon
        gap_outer_bound =0
        self.result={"gap_inner_bound": gap_inner_bound,
                      "gap_outer_bound": gap_outer_bound,
                      "Gbar": Gbar,
                      "std": s,
                      "Glist": G}
        return(self.result)
        

if __name__ == "__main__":
# To test: python mmw_ci.py --num-scens=3  --MMW-num-batches=3 --MMW-batch-size=3
    
    refmodel = "mpisppy.tests.examples.farmer" #Change this path to use a different model
    #Compute the nonant xhat (the one used in the left term of MMW (9) ) using
    #                                                        the first scenarios
    
    ama_options = {"EF-2stage": True,# 2stage vs. mstage
               "start": False}   #Are the scenario shifted by a start arg ?
    
    #add information about batches
    ama_extraargs = argparse.ArgumentParser(add_help=False)
    ama_extraargs.add_argument("--MMW-num-batches",
                            help="number of batches used for MMW confidence interval (default 1)",
                            dest="num_batches",
                            type=int,
                            default=1)
    
    ama_extraargs.add_argument("--MMW-batch-size",
                            help="batch size used for MMW confidence interval (default None)",
                            dest="batch_size",
                            type=int,
                            default=None) #None means take batch_size=num_scens
    
    ama_object = ama.from_module(refmodel, ama_options,extraargs=ama_extraargs)
    ama_object.run()
    
    if global_rank==0 :
        print("inner bound=", ama_object.best_inner_bound)
        # This the xhat of the left term of LHS of MMW (9)
        print("outer bound=", ama_object.best_outer_bound)
    
    
    ########### get the nonants (the xhat)
    nonant_cache = sputils.nonant_cache_from_ef(ama_object.ef)
    ciutils.write_xhat(nonant_cache, path="xhat.npy")
    
    #Set parameters for run()

    options = ama_object.options
    options['solver_options'] = options['EF_solver_options']
    xhat = ciutils.read_xhat("xhat.npy")
   
    
    num_batches = ama_object.options['num_batches']
    batch_size = ama_object.options['batch_size']
    
    mmw = MMWConfidenceIntervals(refmodel, options, xhat, num_batches,batch_size=batch_size,
                       verbose=False)
    r=mmw.run()
    global_toc(r)
    if global_rank==0:
        os.remove("xhat.npy") 
    
