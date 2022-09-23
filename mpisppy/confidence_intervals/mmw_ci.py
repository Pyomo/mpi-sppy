# This software is distributed under the 3-clause BSD License.

# Code to evaluate a given x-hat given as a nonant-cache, and the MMW confidence interval.
# To test: python mmw_ci.py --num-scens=3  --MMW-num-batches=3 --MMW-batch-size=3
# or: python3 mmw_ci.py --num-scens=3  --MMW-num-batches=3 --MMW-batch-size=3 --EF-solver-name cplex

import mpisppy.MPI as mpi
import argparse
import numpy as np
import scipy.stats
import importlib
import os
from mpisppy import global_toc
    
fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

import mpisppy.utils.amalgamator as ama
import mpisppy.utils.xhat_eval as xhat_eval
import mpisppy.utils.sputils as sputils
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
        cfg (Config): useful options to run amalgamator or xhat_eval, 
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
        cfg can include the following things:
            -The type of our solving. for now, MMWci only support EF, so it must
            have an attribute 'EF-2stage' or 'EF-mstage' set equal to True
            - Solver-related options ('EF_solver_name' and 'EF_solver_options')
            
    
    """
    def __init__(self,
                 refmodel,
                 cfg,
                 xhat_one,
                 num_batches,
                 batch_size=None,
                 start=None,
                 verbose=True,
                 ):
        self.refmodel = importlib.import_module(refmodel)
        self.refmodelname = refmodel
        self.cfg = cfg
        self.xhat_one = xhat_one
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.verbose = verbose

        #Getting the start
        if start is None :
            raise RuntimeError( "Start must be specified")
        self.start = start
            
        #Type of our problem
        if ama._bool_option(cfg, "EF_2stage"):
            self.type = "EF_2stage"
            self.multistage = False
            self.numstages = 2
        elif ama._bool_option(cfg, "EF_mstage"):
            self.type = "EF_mstage"
            self.multistage = True
            self.numstages = len(cfg['branching_factors'])+1
        else:
            raise RuntimeError("Only EF is currently supported; "
                "cfg should have an attribute 'EF-2stage' or 'EF-mstage' set to True")
        
        #Check if refmodel and args have all needed attributes
        everything = ["scenario_names_creator", "scenario_creator", "kw_creator"]
        if self.multistage:
            everything[0] = "sample_tree_scen_creator"
    
        you_can_have_it_all = True
        for ething in everything:
            if not hasattr(self.refmodel, ething):
                print(f"Module {refmodel} is missing {ething}")
                you_can_have_it_all = False
        if not you_can_have_it_all:
            raise RuntimeError(f"Module {refmodel} not complete for MMW")
        
        if "EF_solver_name" not in self.cfg:
            raise RuntimeError("EF_solver_name not in Argument list for MMW")

    def run(self, confidence_level=0.95, objective_gap=False):

        # We get the MMW right term, then xhat, then the MMW left term.

        #Compute the nonant xhat (the one used in the left term of MMW (9) ) using
        #                                                        the first scenarios
        
        ############### get the parameters
        start = self.start
        scenario_denouement = self.refmodel.scenario_denouement

        #Introducing batches otpions
        num_batches = self.num_batches
        batch_size = self.batch_size
        sample_cfg = self.cfg()  # ephemeral changes
        
        #Some options are specific to 2-stage or multi-stage problems
        if self.multistage:
            sampling_branching_factors = ciutils.branching_factors_from_numscens(batch_size,self.numstages)
            #TODO: Change this to get a more logical way to compute branching_factors
            batch_size = np.prod(sampling_branching_factors)
        else:
            sampling_BFs = None
            if batch_size == 0:
                raise RuntimeError("batch size can't be zero for two stage problems")
        sample_cfg.quick_assign('num_scens', int, batch_size)
        sample_cfg.quick_assign('_mpisppy_probability', float, 1/batch_size)
        scenario_creator_kwargs=self.refmodel.kw_creator(sample_cfg)
        sample_scen_creator = self.refmodel.scenario_creator
        
        #Solver settings
        solver_name = self.cfg['EF_solver_name']
        solver_options = self.cfg.get('EF_solver_options')
        solver_options = remove_None(solver_options)
            
        #Now we compute for each batch the whole Gn term from MMW (9)

        G = np.zeros(num_batches) #the Gbar of MMW (10)
        #we will compute the mean via a loop (to be parallelized ?)
        zhats = [] #evaluation of xhat at each scenario
        zstars=[]
        for i in range(num_batches) :
            scenstart = None if self.multistage else start
            gap_options = {'seed':start,'branching_factors':sampling_branching_factors} if self.multistage else None
            scenario_names = self.refmodel.scenario_names_creator(batch_size,start=scenstart)
            estim = ciutils.gap_estimators(self.xhat_one, self.refmodelname,
                                           solving_type=self.type,
                                           scenario_names=scenario_names,
                                           sample_options=gap_options,
                                           ArRP=1,
                                           cfg=sample_cfg,
                                           scenario_denouement=scenario_denouement,
                                           solver_name=solver_name,
                                           solver_options=solver_options,
                                           objective_gap=objective_gap)
            Gn = estim['G']
            start = estim['seed']

            # collect evaluation of xhat at all scenario
            if objective_gap:
                for zhat in estim['zhats']:
                    zhats.append(zhat)
                for zstar in estim['zstars']:
                    zstars.append(zstar)      

            if(self.verbose):
                global_toc(f"Gn={Gn} for the batch {i}")  # Left term of LHS of (9)
            G[i]=Gn 

        s_g = np.std(G) #Standard deviation of gap

        Gbar = np.mean(G)

        t_g = scipy.stats.t.ppf(confidence_level,num_batches-1)

        epsilon_g = t_g*s_g/np.sqrt(num_batches)

        gap_inner_bound =  Gbar + epsilon_g
        gap_outer_bound = 0
 
        if objective_gap == True:
            # find confidence interval for zhat
            zhat_bar = np.mean(zhats)
            s_zhat = np.std(zhats)
            t_zhat = scipy.stats.t.ppf(confidence_level, len(zhats)-1)
            epsilon_zhat = t_zhat*s_zhat / np.sqrt(len(zhats))

            zstar_bar = np.mean(zstars)
            s_zstar = np.std(zstars)

            # compute conservative interval on zhat plus 
            # optimality gap in which we expect zstar to lie
            obj_upper_bound = zhat_bar + epsilon_zhat
            obj_lower_bound = zhat_bar - epsilon_zhat - (Gbar + epsilon_g)

        self.result={"gap_inner_bound": gap_inner_bound,
                      "gap_outer_bound": gap_outer_bound,
                      "Gbar": Gbar,
                      "std": s_g,
                      "Glist": G}

        if objective_gap:
            self.result["obj_upper_bound"] = obj_upper_bound
            self.result["obj_lower_bound"] = obj_lower_bound
            self.result["zhat_bar"] = zhat_bar
            self.result["std_zhat"] = s_zhat
            self.result["zstar_bar"] = zstar_bar
            self.result["std_zstar"] = s_zstar

        return(self.result)
        
        
if __name__ == "__main__":
    raise RuntimeError("mmw_ci does not have a __main__")
