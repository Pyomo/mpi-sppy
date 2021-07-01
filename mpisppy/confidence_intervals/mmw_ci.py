# Code to evaluate a given x-hat given as a nonant-cache, and the MMW confidence interval.
# To test: python mmw_ci.py --num-scens=3  --MMW-num-batches=3 --MMW-batch-size=3


import mpi4py.MPI as mpi
#import mpisppy.utils.sputils as sputils
#import mpisppy.utils.xhat_eval as xhat_eval
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

def remove_None(d):
    if d is None:
        return {}
    d_copy = {}
    for (key,value) in d.items():
        if value is not None:
            d_copy[key] = value
    return d_copy

def correcting_numeric(G,relative_error=True,threshold=10**(-4),objfct=None):
    #Correcting small negative of G due to numerical error while solving EF 
    if relative_error:
        if objfct is None:
            raise RuntimeError("We need a value of the objective function to remove numerically negative G")
        elif (G<= -threshold*np.abs(objfct)):
            print("We compute a gap estimator that is anormaly negative")
            return G
        else:
            return max(0,G)
    else:
        if (G<=-threshold):
            raise RuntimeWarning("We compute a gap estimator that is anormaly negative")
            return G
        else: 
            return max(0,G)           
        
 
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
                

class MMWConfidenceIntervals():
    """Takes a model and options as input. 
    Args:
        refmodel (str): path of the model we use (e.g. farmer, uc)
        options (dict): useful options to run amalgomator or xhat_eval, 
                        including EF_solver_options and EF_solver_name
                        May include the options used to compute xhat
        xhat (dict): Non-anticipative solution, computed before
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
                 xhat,
                 num_batches,
                 batch_size=None,
                 start=None,
                 verbose=True,
                 ):
        self.refmodel = importlib.import_module(refmodel)
        self.refmodelname = refmodel
        self.options = options
        self.xhat = xhat
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.verbose = verbose
        
        #Check if refmodel and args have all needed attributes
        
        everything = ["scenario_names_creator",
                 "scenario_creator",
                 "kw_creator"]  # denouement can be missing.
    
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
                "Scenarios used to compute xhat may be used in MMW")
            
        #Type of our problem
        if ama._bool_option(options, "EF-2stage"):
            self.type = "EF-2stage" 
        elif ama._bool_option(options, "EF-mstage"):
            self.type = "EF-mstage"
        else:
            raise RuntimeError(
                "Only EF is supported. options should get an attribute 'EF-2stage' or 'EF-mstage' set to True")
        
        if self.type == "EF-mstage" and "BFs" not in options:
            raise RuntimeError(
                "For multi-stage problems, we need branching factors (an attribute 'BFs' to options)")
        
    def run(self,confidence_level=0.95):
        # We get the MMW right term, then xhat, then the MMW left term.


        #Compute the nonant xhat (the one used in the left term of MMW (9) ) using
        #                                                        the first scenarios
        ########### get the nonants (the xhat)
        xhat = self.xhat
        
        ############### get the parameters
        ScenCount = self.start
        scenario_creator = self.refmodel.scenario_creator
        scenario_denouement = self.refmodel.scenario_denouement

        
        #Introducing batches otpions
        num_batches = self.num_batches
        bs=self.batch_size
        batch_size = bs if (bs is not None) else ScenCount #is None : take size_batch=num_scens
        scenario_creator_kwargs=self.refmodel.kw_creator(self.options)
        scenario_creator_kwargs['num_scens'] = batch_size
        solvername = self.options['EF_solver_name']
        solver_options = self.options['EF_solver_options'] if 'EF_solver_options' in self.options else None
        solver_options = remove_None(solver_options)
            
        #Now we compute for each batch the whole Gn term from MMW (9)

        
        G = np.zeros(num_batches) #the Gbar of MMW (10)
        #we will compute the mean via a loop (to be parallelized ?)
        
        
        for i in range(num_batches) :
            #First we compute the right term of MMW (9)
            start = ScenCount+i*batch_size
            MMW_scenario_names = self.refmodel.scenario_names_creator(
                batch_size, start=start)
            

            
            #We use amalgomator to do it

            ama_options = dict(scenario_creator_kwargs)
            ama_options['start'] = start
            ama_options['EF_solver_name'] = solvername
            ama_options[self.type] = True
            ama_object = ama.from_module(self.refmodelname, ama_options,use_command_line=False)
            ama_object.verbose = self.verbose
            ama_object.run()
            MMW_right_term = ama_object.EF_Obj
            
            #Then we compute the left term of (9)
            # Create the eval object for the left term of the LHS of (9) in MMW
            
            options = {"iter0_solver_options": None,
                     "iterk_solver_options": None,
                     "display_timing": False,
                     "solvername": solvername,
                     "verbose": False,
                     "solver_options":solver_options}
            
            if self.type == "EF-mstage":
                all_nodenames = sputils.create_nodenames_from_BFs(
                    self.options['BFs'])
                options['branching_factors'] = self.options['BFs']
            else:
                all_nodenames = None
                
            ev = xhat_eval.Xhat_Eval(options,
                            MMW_scenario_names,
                            scenario_creator,
                            scenario_denouement,
                            scenario_creator_kwargs=scenario_creator_kwargs,
                            all_nodenames = all_nodenames)
            obj_at_xhat = ev.evaluate(xhat)

            #Now we can compute MMW (9)
            Gn = obj_at_xhat-MMW_right_term
            use_relative_error = (np.abs(MMW_right_term)>1)
            Gn = correcting_numeric(Gn,
                                    relative_error=use_relative_error,
                                    objfct=MMW_right_term)
            
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
        

class MMWObjectiveConfidenceIntervals():
    """Takes a model and options as input. 
    Args:
        refmodel (str): path of the model we use (e.g. farmer, uc)
        options (dict): useful options to run amalgomator or xhat_eval, 
                        including EF_solver_options and EF_solver_name
                        May include the options used to compute xhat
        xhat (dict): Non-anticipative solution, computed before
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
                 xhat,
                 num_batches,
                 batch_size=None,
                 start=None,
                 verbose=True,
                 ):
        self.refmodel = importlib.import_module(refmodel)
        self.refmodelname = refmodel
        self.options = options
        self.xhat = xhat
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.verbose = verbose
        
        #Check if refmodel and args have all needed attributes
        
        everything = ["scenario_names_creator",
                 "scenario_creator",
                 "kw_creator"]  # denouement can be missing.
    
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
                "Scenarios used to compute xhat may be used in MMW")
            
        #Type of our problem
        if ama._bool_option(options, "EF-2stage"):
            self.type = "EF-2stage" 
        elif ama._bool_option(options, "EF-mstage"):
            self.type = "EF-mstage"
        else:
            raise RuntimeError(
                "Only EF is supported. options should get an attribute 'EF-2stage' or 'EF-mstage' set to True")
        
        if self.type == "EF-mstage" and "BFs" not in options:
            raise RuntimeError(
                "For multi-stage problems, we need branching factors (an attribute 'BFs' to options)")
        
    def run(self,confidence_level=0.95):
        # We get the MMW right term, then xhat, then the MMW left term.


        #Compute the nonant xhat (the one used in the left term of MMW (9) ) using
        #                                                        the first scenarios
        ########### get the nonants (the xhat)
        xhat = self.xhat
        
        ############### get the parameters
        ScenCount = self.start
        scenario_creator = self.refmodel.scenario_creator
        scenario_denouement = self.refmodel.scenario_denouement

        
        #Introducing batches otpions
        num_batches = self.num_batches
        bs=self.batch_size
        batch_size = bs if (bs is not None) else ScenCount #is None : take size_batch=num_scens
        scenario_creator_kwargs=self.refmodel.kw_creator(self.options)
        scenario_creator_kwargs['num_scens'] = batch_size
        solvername = self.options['EF_solver_name']
        solver_options = self.options['EF_solver_options'] if 'EF_solver_options' in self.options else None
        solver_options = remove_None(solver_options)
            
        #Now we compute for each batch the whole Gn term from MMW (9)

        
        G = np.zeros(num_batches) #the Gbar of MMW (10)
        #we will compute the mean via a loop (to be parallelized ?)
        objectives = []
        
        for i in range(num_batches) :
            #First we compute the right term of MMW (9)
            start = ScenCount+i*batch_size
            MMW_scenario_names = self.refmodel.scenario_names_creator(
                batch_size, start=start)
            

            
            #We use amalgomator to do it

            ama_options = dict(scenario_creator_kwargs)
            ama_options['start'] = start
            ama_options['EF_solver_name'] = solvername
            ama_options[self.type] = True
            ama_object = ama.from_module(self.refmodelname, ama_options,use_command_line=False)
            ama_object.verbose = self.verbose
            ama_object.run()
            MMW_right_term = ama_object.EF_Obj
            objectives.append(MMW_right_term)
            
            #Then we compute the left term of (9)
            # Create the eval object for the left term of the LHS of (9) in MMW
            
            options = {"iter0_solver_options": None,
                     "iterk_solver_options": None,
                     "display_timing": False,
                     "solvername": solvername,
                     "verbose": False,
                     "solver_options":solver_options}
            
            if self.type == "EF-mstage":
                all_nodenames = sputils.create_nodenames_from_BFs(
                    self.options['BFs'])
                options['branching_factors'] = self.options['BFs']
            else:
                all_nodenames = None
                
            ev = xhat_eval.Xhat_Eval(options,
                            MMW_scenario_names,
                            scenario_creator,
                            scenario_denouement,
                            scenario_creator_kwargs=scenario_creator_kwargs,
                            all_nodenames = all_nodenames)
            obj_at_xhat = ev.evaluate(xhat)

            #Now we can compute MMW (9)
            Gn = obj_at_xhat-MMW_right_term
            use_relative_error = (np.abs(MMW_right_term)>1)
            Gn = correcting_numeric(Gn,
                                    relative_error=use_relative_error,
                                    objfct=MMW_right_term)
            
            if(self.verbose):
                global_toc(f"Gn={Gn} for the batch {i}")  # Left term of LHS of (9)
            G[i]=Gn             
        s = np.std(G) #Standard deviation
        Gbar = np.mean(G)
        zbar = np.mean(objectives)
        t = scipy.stats.t.ppf(confidence_level,num_batches-1)
        epsilon = t*s/np.sqrt(num_batches)
        gap_inner_bound =  zbar + Gbar+epsilon
        gap_outer_bound = zbar
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
    write_xhat(nonant_cache, path="xhat.npy")
    
    #Set parameters for run()

    options = ama_object.options
    options['solver_options'] = options['EF_solver_options']
    xhat = read_xhat("xhat.npy")
   
    
    num_batches = ama_object.options['num_batches']
    batch_size = ama_object.options['batch_size']
    
    mmw = MMWConfidenceIntervals(refmodel, options, xhat, num_batches,batch_size=batch_size,
                       verbose=False)
    r=mmw.run()
    global_toc(r)
    if global_rank==0:
        os.remove("xhat.npy") 