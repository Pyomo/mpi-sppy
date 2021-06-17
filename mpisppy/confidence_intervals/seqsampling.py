# Copyright 2021 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Code that is producing a xhat and a confidence interval using sequantial sampling 
# This is the implementation of the 2 following papers:
# [bm2011] Bayraksan, G., Morton,D.P.: A Sequential Sampling Procedure for Stochastic Programming. Operations Research 59(4), 898-913 (2011)
# [bpl2012] Bayraksan, G., Pierre-Louis, P.: Fixed-Width Sequential Stopping Rules for a Class of Stochastic Programs, SIAM Journal on Optimization 22(4), 1518-1548 (2012)


import pyomo.environ as pyo
import mpi4py.MPI as mpi
import mpisppy.utils.sputils as sputils
import numpy as np
import scipy.stats
import importlib
from mpisppy import global_toc
    
fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

import mpisppy.utils.amalgomator as amalgomator
import mpisppy.utils.xhat_eval as xhat_eval
import mpisppy.confidence_intervals.mmw_ci as MMWci
from mpisppy.tests.examples.apl1p import xhat_generator_apl1p

#==========


def is_needed(options,needed_things,message=""):
    if not set(needed_things)<= set(options):
        print("Some options are missing. "+message)
        

def add_options(options,optional_things,optional_default_settings):
    for i in range(len(optional_things)):
        ething = optional_things[i]
        if not ething in options :
            options[ething]=optional_default_settings[i]

def xhat_generator_farmer(scenario_names, solvername="gurobi", solver_options=None, crops_multiplier=1):
    '''Farmer example applied to sequential sampling. Given scenario names and
    options, create the scenarios and compute the xhat that is minimizing the
    approximate probleme associatd with these scenarios.

    Parameters
    ----------
    scenario_names: int
        Names of the scenario we use
    solvername: str, optional
        Name of the solver used. The default is "gurobi".
    solver_options: dict, optional
        Solving options. The default is None.
    crops_multiplier: int, optional
        A parameter of the farmer model. The default is 1.

    Returns
    -------
    xhat: xhat object (dict containing a 'ROOT' key with a np.array)
        A generated xhat.

    '''
    num_scens = len(scenario_names)
    
    ama_options = { "EF-2stage": True,
                    "EF_solver_name": solvername,
                    "EF_solver_options": solver_options,
                    "use_integer": False,
                    "crops_multiplier": crops_multiplier,
                    "num_scens": num_scens,
                    "_mpisppy_probability": 1/num_scens,
                    }
    #We use from_module to build easily an Amalgomator object
    ama = amalgomator.from_module("mpisppy.tests.examples.farmer",
                                  ama_options,use_command_line=False)
    #Correcting the building by putting the right scenarios.
    ama.scenario_names = scenario_names
    ama.run()
    
    # get the xhat
    xhat = sputils.nonant_cache_from_ef(ama.ef)

    return xhat

def gap_estimators(xhat,solvername, scenario_names, scenario_creator, ArRP=1,
                   scenario_creator_kwargs={}, scenario_denouement=None,
                   solver_options=None):
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

    Returns
    -------
    G_k and s_k, gap estimator and associated standard deviation estimator.

    '''
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
                                       solver_options=solver_options)
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


class SeqSampling():
    """
    Computing a solution xhat and a confidence interval for the optimality gap sequentially,
    by taking an increasing number of scenarios.
    
    Args:
        refmodel (str): path of the model we use (e.g. farmer, uc)
        xhat_generator (function): a function that takes scenario_names (and 
                                    and optional solvername and solver_options) as input
                                    and return a xhat.

        options (dict): multiple useful parameters, e.g.:
                        - "solvername", str, the name of the solver we use 
                            (default is gurobi)
                        - "solver_options", dict containing solver options 
                            (default is {}, an empty dict)
                        - "sample_size_ratio", float, the ratio (xhat sample size)/(gap estimators sample size)
                            (default is 1)
                        - "xhat_gen_options" dict containing options passed to the xhat generator
                            (default is {}, an empty dict)
                        - "ArRP", int, how much estimators should be pooled to compute G and s ?
                            (default is 1, no pooling)
                        - "kf_Gs", int, resampling frequency to compute estimators
                            (default is 1, always resample completely)
                        - "kf_xhat", int, resampling frequency to compute xhat
                            (default is 1, always resample completely)
                        -"confidence_level", float, asymptotic confidence level 
                            of the output confidence interval
                            (default is 0.95)
                        -Some other parameters, depending on what model 
                            (BM or BPL, deterministic or sequential sampling)
                        
        stochastic_sampling (bool, default False):  should we compute sample sizes using estimators ?
            if stochastic_sampling is True, we compute sample size using §5 of [Bayraksan and Pierre-Louis]
            else, we compute them using [Bayraksan and Morton] technique
        stopping_criterion (str, default 'BM'): which stopping criterion should be used ?
            2 criterions are supported : 'BM' for [Bayraksan and Morton] and 'BPL' for [Bayraksan and Pierre-Louis]
    """
    
    def __init__(self,
                 refmodel,
                 xhat_generator,
                 options,
                 stochastic_sampling = False,
                 stopping_criterion = "BM"):
        
        self.refmodel = importlib.import_module(refmodel)
        self.refmodelname = refmodel
        self.scenario_creator_kwargs = self.refmodel.kw_creator(options)
        self.xhat_generator = xhat_generator
        self.options = options
        self.stochastic_sampling = stochastic_sampling
        self.stopping_criterion = stopping_criterion
        self.solvername = options["solvername"] if "solvername" in options else "gurobi"
        self.solver_options = options["solver_options"] if "solver_options" in options else None
        self.sample_size_ratio = options["sample_size_ratio"] if "sample_size_ration" in options else 1
        self.xhat_gen_options = options["xhat_gen_options"] if "xhat_gen_options" in options else {}
        
        
        
        #Check if refmodel has all needed attributes
        everything = ["scenario_names_creator",
                 "scenario_creator",
                 "kw_creator"]  # denouement can be missing.
        you_can_have_it_all = True
        for ething in everything:
            if not hasattr(self.refmodel, ething):
                print(f"Module {refmodel} is missing {ething}")
                you_can_have_it_all = False
        if not you_can_have_it_all:
            raise RuntimeError(f"Module {refmodel} not complete for seqsampling")
            
        #Manage options
        optional_things = ["ArRP","kf_Gs","kf_xhat","confidence_level"]
        optional_default_settings = [1,1,1,0.95]
        add_options(options, optional_things, optional_default_settings)      
        
        if self.stochastic_sampling :
                add_options(options, ["n0min"], [50])
                
                
        if self.stopping_criterion == "BM":
            needed_things = ["epsprime","hprime","eps","h","p"]
            is_needed(options, needed_things)
            optional_things = ["q"]
            optional_default_settings = [None]
            add_options(options, optional_things, optional_default_settings)
        elif self.stopping_criterion == "BPL":
            is_needed(options, ["eps"])
            if not self.stochastic_sampling :
                optional_things = ["c0","c1","growth_function"]
                optional_default_settings = [50,2,(lambda x : x-1)]
                add_options(options, optional_things, optional_default_settings)
        else:
            raise RuntimeError("Only BM and BPL criteria are supported yet")
        for oname in options:
            setattr(self, oname, options[oname]) #Set every option as an attribute
        
        #To be sure to always use new scenarios, we set a ScenCount that is 
        #telling us how many scenarios has been used so far
        self.ScenCount = 0
            
    def bm_stopping_criterion(self,G,s,nk):
        return(G>self.hprime*s+self.epsprime)
    
    def bpl_stopping_criterion(self,G,s,nk):
        t = scipy.stats.t.ppf(self.confidence_level,nk-1)
        sample_error = t*s/np.sqrt(nk)
        inflation_factor = 1/np.sqrt(nk)
        return(G+sample_error+inflation_factor>self.eps)
    
    def bm_sampsize(self,k,G,s,nk_m1, r=2):
        h = self.h
        hprime = self.hprime
        p = self.p
        q = self.q
        confidence_level = self.confidence_level
        if q is None :
            # Computing n_k as in (5) of [Bayraksan and Morton, 2009]
            if hasattr(self, "c") :
                c = self.c
            else:
                if confidence_level is None :
                    raise RuntimeError("We need the confidence level to compute the constant cp")
                j = np.arange(1,1000)
                s = sum(np.power(j,-p*np.log(j)))
                c = max(1,2*np.log(s/(np.sqrt(2*np.pi)*(1-confidence_level))))
            
            lower_bound = (c+2*p* np.log(k)**2)/((h-hprime)**2)
        else :
            # Computing n_k as in (14) of [Bayraksan and Morton, 2009]
            if hasattr(self, "c") :
                c = self.c
            else:
                if confidence_level is None :
                    RuntimeError("We need the confidence level to compute the constant c_pq")
                j = np.arange(1,1000)
                s = sum(np.exp(-p*np.power(j,2*q/r)))
                c = max(1,2*np.log(s/(np.sqrt(2*np.pi)*(1-confidence_level))))
            
            lower_bound = (c+2*p*np.power(k,2*q/r))/((h-hprime)**2)   
        #print(f"nk={lower_bound}")
        return int(np.ceil(lower_bound))
    
    def bpl_fsp_sampsize(self,k,G,s,nk_m1):
        return(int(np.ceil(self.c0+self.c1*self.growth_function(k))))
        
    def stochastic_sampsize(self,k,G,s,nk_m1):
        if (k==1):
            #Initialization
            return(int(np.ceil(max(self.n0min,np.log(1/self.eps)))))
        #§5 of [Bayraksan and Pierre-Louis] : solving a 2nd degree equation in sqrt(n)
        t = scipy.stats.t.ppf(self.confidence_level,nk_m1-1)
        a = - self.eps
        b = 1+t*s
        c = nk_m1*G
        maxroot = -(np.sqrt(b**2-4*a*c)+b)/(2*a)
        print(f"s={s}, t={t}, G={G}")
        print(f"a={a}, b={b},c={c},delta={b**2-4*a*c}")
        print(f"At iteration {k}, we took n_k={int(np.ceil((maxroot**2)))}")
        return(int(np.ceil(maxroot**2)))
    
    
    def run(self,maxit=200):
        refmodel = self.refmodel
        mult = self.sample_size_ratio # used to set m_k= mult*n_k
        
        #Get the stopping criterion
        if self.stopping_criterion == "BM":
            stop_criterion = self.bm_stopping_criterion
        elif self.stopping_criterion == "BPL":
            stop_criterion = self.bpl_stopping_criterion
        else:
            raise RuntimeError("Only BM and BPL criteria are supported yet")
            
        #Get the function computing sample size
        if self.stochastic_sampling:
            self.sample_size = self.stochastic_sampsize
        elif self.stopping_criterion == "BM":
            self.sample_size = self.bm_sampsize
        elif self.stopping_criterion == "BPL":
            self.sample_size = self.bpl_fsp_sampsize
        else:
            raise RuntimeError("Only BM and BPL sample sizes are supported yet")
        
        #----------------------------Step 0 -------------------------------------#
        #Initialization
        k =1
        
        
        #Computing n_1


        if self.stopping_criterion == "BM":
            #Finding a constant used to compute nk
            r = 2 #TODO : we could add flexibility here
            j = np.arange(1,1000)
            if self.q is None:
                s = sum(np.power(j,-self.p*np.log(j)))
            else:
                if self.q<1:
                    raise RuntimeError("Parameter q should be greater than 1.")
                s = sum(np.exp(-self.p*np.power(j,2*self.q/r)))
            self.c = max(1,2*np.log(s/(np.sqrt(2*np.pi)*(1-self.confidence_level))))
                
        
        nk = self.ArRP *int(np.ceil(self.sample_size(k, None, None, None)/self.ArRP))
        
        #Computing xhat_1. 
        #We use sample_size_ratio*n_k observations to compute xhat_k
        xhat_scenario_names = refmodel.scenario_names_creator(mult*nk, start=0)
        self.ScenCount+=mult*nk   
        
        xhat_k = self.xhat_generator(xhat_scenario_names,
                                   solvername=self.solvername,
                                   solver_options=self.solver_options,
                                   **self.xhat_gen_options)
        #Sample observations used to compute G_k and s_k
        estimator_scenario_names = refmodel.scenario_names_creator(nk,
                                                                   start=self.ScenCount)
        self.ScenCount+=nk
    
        #----------------------------Step 1 -------------------------------------#
        
        #Computing G_nkand s_k associated with xhat_1
        scenario_creator_kwargs = self.scenario_creator_kwargs if hasattr(self, "scenario_creator_kwargs") else {}
        scenario_denouement = refmodel.scenario_denouement if hasattr(refmodel, "scenario_denouement") else None
        Gk,sk = gap_estimators(xhat_k,
                               self.solvername, 
                               estimator_scenario_names, 
                               refmodel.scenario_creator, ArRP=self.ArRP,
                               scenario_creator_kwargs=scenario_creator_kwargs,
                               scenario_denouement=scenario_denouement,
                               solver_options=self.solver_options)
        
        #----------------------------Step 2 -------------------------------------#

        while( stop_criterion(Gk,sk,nk) and k<maxit):
        #----------------------------Step 3 -------------------------------------#       
            k+=1
            nk_m1 = nk #n_{k-1}
            nk = self.ArRP*int(np.ceil(self.sample_size(k, Gk, sk, nk_m1)/self.ArRP))
            assert nk>= nk_m1, "Our sample size should be increasing"
            #Computing xhat_k
            if (k%self.kf_xhat==0):
                #We use only new scenarios to compute xhat
                xhat_scenario_names = refmodel.scenario_names_creator(mult*nk,
                                                                      start=self.ScenCount)
                self.ScenCount+=mult*nk
            else:
                #We reuse the previous scenarios
                xhat_scenario_names+= refmodel.scenario_names_creator(mult*(nk-nk_m1),
                                                                      start=self.ScenCount)
                self.ScenCount+= mult*(nk-nk_m1)
            xhat_k = self.xhat_generator(xhat_scenario_names,
                                       solvername=self.solvername,
                                       solver_options=self.solver_options,
                                       **self.xhat_gen_options)
            #Computing G_k and s_k
            if (k%self.kf_Gs==0):
                #We use only new scenarios to compute xhat
                estimator_scenario_names = refmodel.scenario_names_creator(nk,
                                                                           start=self.ScenCount)
                self.ScenCount+=nk
            else:
                #We reuse the previous scenarios
                estimator_scenario_names+= refmodel.scenario_names_creator((nk-nk_m1),
                                                                           start=self.ScenCount)
                self.ScenCount+= (nk-nk_m1)

            Gk,sk = gap_estimators(xhat_k,
                                   self.solvername, 
                                   estimator_scenario_names, 
                                   refmodel.scenario_creator, ArRP=self.ArRP,
                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                   scenario_denouement=scenario_denouement,
                                   solver_options=self.solver_options)

            if (k%10==0):
                print(f"k={k}")
                print(f"n_k={nk}")
        #----------------------------Step 4 -------------------------------------#
        if (k==maxit) :
            raise RuntimeError(f"The loop terminated after {maxit} iteration with no acceptable solution")
        T = k
        final_xhat=xhat_k
        if self.stopping_criterion == "BM":
            upper_bound=self.h*sk+self.eps
        elif self.stopping_criterion == "BPL":
            upper_bound = self.eps
        else:
            raise RuntimeError("Only BM and BPL criterion are supported yet.")
        CI=[0,upper_bound]
        global_toc(f"G={Gk}")
        global_toc(f"s={sk}")
        global_toc(f"xhat has been computed with {nk*mult} observations.")
        return {"T":T,"Candidate_solution":final_xhat,"CI":CI,}

if __name__ == "__main__":
    refmodel = "mpisppy.tests.examples.farmer"
    farmer_opt_dict = {"crops_multiplier":3}
    
    
    optionsBM = {'h':0.2,
               'hprime':0.015, 
              'eps':0.5, 
               'epsprime':0.4, 
               "p":0.2,
               "q":1.2,
               "solvername":"gurobi_direct",
               }
    optionsFSP = {'eps': 5.0,
                  'solvername': "gurobi_direct",
                  "c0":50,}

    optionsSSP = {'eps': 1.0,
                  'solvername': "gurobi_direct",    
                  "n0min":200,
                  #"xhat_gen_options": farmer_opt_dict,
                  #"crops_multiplier": 3,
                  }
    our_pb = SeqSampling(refmodel,
                          xhat_generator_farmer,
                          optionsBM,
                          stochastic_sampling=False,
                          stopping_criterion="BM",
                          )
    res = our_pb.run()

    print(res)        