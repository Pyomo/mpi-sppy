# Copyright 2021 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Code that is producing a xhat and a confidence interval using sequential sampling 
# This is the implementation of the 2 following papers:
# [bm2011] Bayraksan, G., Morton,D.P.: A Sequential Sampling Procedure for Stochastic Programming. Operations Research 59(4), 898-913 (2011)
# [bpl2012] Bayraksan, G., Pierre-Louis, P.: Fixed-Width Sequential Stopping Rules for a Class of Stochastic Programs, SIAM Journal on Optimization 22(4), 1518-1548 (2012)

# see also multi_seqsampling.py, which has a class derived from this class

import pyomo.environ as pyo
import mpisppy.MPI as mpi
import mpisppy.utils.sputils as sputils
import numpy as np
import scipy.stats
import importlib
from mpisppy import global_toc
from mpisppy.utils import config
    
fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

import mpisppy.utils.amalgamator as amalgamator
import mpisppy.utils.xhat_eval as xhat_eval
import mpisppy.confidence_intervals.ciutils as ciutils
from mpisppy.tests.examples.apl1p import xhat_generator_apl1p

#==========

def is_needed(options,needed_things,message=""):
    if not set(needed_things)<= set(options):
        raise RuntimeError("Some options are missing from this list of reqiored options:\n"
                           f"{needed_things}\n"
                           f"{message}")
        

def add_options(cfg, optional_things):
    # allow for defaults on options that Bayraksan et al establish 
    for i,v  in optional_things.items():
        if not i in cfg:
            print(f"{type(v) =}")
            cfg.quick_assign(i, float, v)

def xhat_generator_farmer(scenario_names, solvername="gurobi", solver_options=None, crops_multiplier=1):
    ''' For developer testing: Given scenario names and
    options, create the scenarios and compute the xhat that is minimizing the
    approximate problem associated with these scenarios.

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

    NOTE: this is here for testing during development.

    '''
    num_scens = len(scenario_names)
    
    cfg = config.Config()
    cfg.quick_assign("EF_2stage", bool, True)
    cfg.quick_assign("EF_solver_name", str, solvername)
    cfg.quick_assign("EF_solver_options", dict, solver_options)
    cfg.quick_assign("num_scens", int, num_scens)
    cfg.quick_assign("_mpisppy_probability", float, 1/num_scens)
    cfg.quick_assign("start_seed", int, start_seed)

    #We use from_module to build easily an Amalgamator object
    ama = amalgamator.from_module("mpisppy.tests.examples.farmer",
                                  cfg, use_command_line=False)
    #Correcting the building by putting the right scenarios.
    ama.scenario_names = scenario_names
    ama.run()
    
    # get the xhat
    xhat = sputils.nonant_cache_from_ef(ama.ef)

    return xhat


class SeqSampling():
    """
    Computing a solution xhat and a confidence interval for the optimality gap sequentially,
    by taking an increasing number of scenarios.
    
    Args:
        refmodel (str): path of the model we use (e.g. farmer, uc)
        xhat_generator (function): a function that takes scenario_names (and 
                                    and optional solvername and solver_options) 
                                    as input and returns a first stage policy 
                                    xhat.

        cfg (Config): multiple parameters, e.g.:
                        - "solvername", str, the name of the solver we use
                        - "solver_options", dict containing solver options 
                            (default is {}, an empty dict)
                        - "sample_size_ratio", float, the ratio (xhat sample size)/(gap estimators sample size)
                            (default is 1)
                        - "xhat_gen_options" dict containing options passed to the xhat generator
                            (default is {}, an empty dict)
                        - "ArRP", int, how many estimators should be pooled to compute G and s ?
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
        solving_type (str, default 'EF-2stage'): how do we solve the approximate problems ?
            Must be one of 'EF-2stage' and 'EF-mstage' (for problems with more than 2 stages).
            Solving methods outside EF are not supported yet.
    """
    
    def __init__(self,
                 refmodel,
                 xhat_generator,
                 cfg,
                 stochastic_sampling = False,
                 stopping_criterion = "BM",
                 solving_type = "None"):
        
        if not isinstance(cfg, config.Config):
            raise RuntimeError(f"SeqSampling bad cfg type={type(cfg)}; should be Config")
        self.refmodel = importlib.import_module(refmodel)
        self.refmodelname = refmodel
        self.xhat_generator = xhat_generator
        self.cfg = cfg
        self.stochastic_sampling = stochastic_sampling
        self.stopping_criterion = stopping_criterion
        self.solving_type = solving_type
        self.solvername = cfg.get("solvername", None)
        self.solver_options = cfg.get("solver_options", None)
        self.sample_size_ratio = cfg.get("sample_size_ratio", 1)
        self.xhat_gen_options = cfg.get("xhat_gen_options", {})
        
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
        optional_options = {"ArRP": 1,
                            "kf_Gs": 1,
                            "kf_xhat": 1,
                            "confidence_level": 0.95}
        add_options(cfg, optional_options)
        
        if self.stochastic_sampling :
                add_options(options, ["n0min"], [50])
                
                
        if self.stopping_criterion == "BM":
            needed_things = ["epsprime","hprime","eps","h","p"]
            is_needed(cfg, needed_things)
            optional_things = {"q": None}
            add_options(cfg, optional_things)
        elif self.stopping_criterion == "BPL":
            is_needed(cfg, ["eps"])
            if not self.stochastic_sampling :
                optional_things = {"c0":50, "c1":2, "growth_function":(lambda x : x-1)}
                add_options(cfg, optional_things)
        else:
            raise RuntimeError("Only BM and BPL criteria are supported at this time.")
        for oname in cfg:
            setattr(self, oname, cfg[oname]) #Set every option as an attribute
        
        #Check the solving_type, and find if the problem is multistage
        two_stage_types = ['EF_2stage']
        multistage_types = ['EF_mstage']
        if self.solving_type in two_stage_types:
            self.multistage = False
        elif self.solving_type in multistage_types:
            self.multistage = True
        else:
            raise RuntimeError(f"The solving_type {self.solving_type} is not supported."
                               f"If you want to run a 2-stage problem, please use a solving_type in {two_stage_types}"
                               f"If you want to run a multistage stage problem, please use a solving_type in {multistage_types}")
        
        #Check the multistage options
        if self.multistage:
            needed_things = ["branching_factors"]
            is_needed(cfg, needed_things)
            if cfg['kf_Gs'] != 1 or cfg['kf_xhat'] != 1:
                raise RuntimeError("Resampling frequencies must be set equal to one for multistage.")
        
        #Get the stopping criterion
        if self.stopping_criterion == "BM":
            self.stop_criterion = self.bm_stopping_criterion
        elif self.stopping_criterion == "BPL":
            self.stop_criterion = self.bpl_stopping_criterion
        else:
            raise RuntimeError("Only BM and BPL criteria are supported.")
            
        #Get the function computing sample size
        if self.stochastic_sampling:
            self.sample_size = self.stochastic_sampsize
        elif self.stopping_criterion == "BM":
            self.sample_size = self.bm_sampsize
        elif self.stopping_criterion == "BPL":
            self.sample_size = self.bpl_fsp_sampsize
        else:
            raise RuntimeError("Only BM and BPL sample sizes are supported yet")
        
        #To be sure to always use new scenarios, we set a ScenCount that is 
        #telling us how many scenarios has been used so far
        self.ScenCount = 0
        
        #If we are running a multistage problem, we also need a seed count
        self.SeedCount = 0
            
    def bm_stopping_criterion(self,G,s,nk):
        # arguments defined in [bm2011]
        return(G>self.hprime*s+self.epsprime)
    
    def bpl_stopping_criterion(self,G,s,nk):
        # arguments defined in [bpl2012]
        t = scipy.stats.t.ppf(self.confidence_level,nk-1)
        sample_error = t*s/np.sqrt(nk)
        inflation_factor = 1/np.sqrt(nk)
        return(G+sample_error+inflation_factor>self.eps)
    
    def bm_sampsize(self,k,G,s,nk_m1, r=2):
        # arguments defined in [bm2011]
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
        # arguments defined in [bpl2012]
        return(int(np.ceil(self.c0+self.c1*self.growth_function(k))))
        
    def stochastic_sampsize(self,k,G,s,nk_m1):
        # arguments defined in [bpl2012]
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
        """ Execute a sequental sampling algorithm
        Args:
            maxit (int): override the stopping criteria based on iterations
        Returns:
            {"T":T,"Candidate_solution":final_xhat,"CI":CI,}
        """
        if self.multistage:
            raise RuntimeWarning("Multistage sequential sampling can be done "
                                 "using the SeqSampling, but dependent samples\n"
                                 "will be used. The class IndepScens_SeqSampling uses independent samples and therefor has better theoretical support.")
        refmodel = self.refmodel
        mult = self.sample_size_ratio # used to set m_k= mult*n_k
        
        
        #----------------------------Step 0 -------------------------------------#
        #Initialization
        k =1
        
        
        #Computing the lower bound for n_1


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
                
        lower_bound_k = self.sample_size(k, None, None, None)
        
        #Computing xhat_1.

        #We use sample_size_ratio*n_k observations to compute xhat_k
        if self.multistage:
            xhat_branching_factors = ciutils.scalable_branching_factors(mult*lower_bound_k, self.options['branching_factors'])
            mk = np.prod(xhat_branching_factors)
            self.xhat_gen_options['start_seed'] = self.SeedCount #TODO: Maybe find a better way to manage seed
            xhat_scenario_names = refmodel.scenario_names_creator(mk)
            
        else:
            mk = int(np.floor(mult*lower_bound_k))
            xhat_scenario_names = refmodel.scenario_names_creator(mk, start=self.ScenCount)
            self.ScenCount+=mk

        xxxxx
        xgo = self.xhat_gen_options.copy()
        xgo.pop("solvername", None)  # it will be given explicitly
        xgo.pop("solver_options", None)  # it will be given explicitly
        xgo.pop("scenario_names", None)  # given explicitly
        xhat_k = self.xhat_generator(xhat_scenario_names,
                                   solvername=self.solvername,
                                   solver_options=self.solver_options,
                                     **xgo)

    
        #----------------------------Step 1 -------------------------------------#
        #Computing n_1 and associated scenario names
        if self.multistage:
            self.SeedCount += sputils.number_of_nodes(xhat_branching_factors)
            
            gap_branching_factors = ciutils.scalable_branching_factors(lower_bound_k, self.options['branching_factors'])
            nk = np.prod(gap_branching_factors)
            estimator_scenario_names = refmodel.scenario_names_creator(nk)
            sample_options = {'branching_factors':gap_branching_factors, 'seed':self.SeedCount}
        else:
            nk = self.ArRP *int(np.ceil(lower_bound_k/self.ArRP))
            estimator_scenario_names = refmodel.scenario_names_creator(nk,
                                                                       start=self.ScenCount)
            sample_options = None
            self.ScenCount+= nk
        
        #Computing G_nkand s_k associated with xhat_1
            
        self.options['num_scens'] = nk
        scenario_denouement = refmodel.scenario_denouement if hasattr(refmodel, "scenario_denouement") else None
        estim = ciutils.gap_estimators(xhat_k, self.refmodelname,
                                       solving_type=self.solving_type,
                                       scenario_names=estimator_scenario_names,
                                       sample_options=sample_options,
                                       ArRP=self.ArRP,
                                       cfg=xxx,
                                       scenario_denouement=scenario_denouement,
                                       solvername=self.solvername,
                                       solver_options=self.solver_options)
        Gk,sk = estim['G'],estim['s']
        if self.multistage:
            self.SeedCount = estim['seed']
        
        #----------------------------Step 2 -------------------------------------#

        while( self.stop_criterion(Gk,sk,nk) and k<maxit):
        #----------------------------Step 3 -------------------------------------#       
            k+=1
            nk_m1 = nk #n_{k-1}
            mk_m1 = mk
            lower_bound_k = self.sample_size(k, Gk, sk, nk_m1)
            
            #Computing m_k and associated scenario names
            if self.multistage:
                xhat_branching_factors = ciutils.scalable_branching_factors(mult*lower_bound_k, self.options['branching_factors'])
                mk = np.prod(xhat_branching_factors)
                self.xhat_gen_options['start_seed'] = self.SeedCount #TODO: Maybe find a better way to manage seed
                xhat_scenario_names = refmodel.scenario_names_creator(mk)
            
            else:
                mk = int(np.floor(mult*lower_bound_k))
                assert mk>= mk_m1, "Our sample size should be increasing"
                if (k%self.kf_xhat==0):
                    #We use only new scenarios to compute xhat
                    xhat_scenario_names = refmodel.scenario_names_creator(int(mult*nk),
                                                                          start=self.ScenCount)
                    self.ScenCount+= mk
                else:
                    #We reuse the previous scenarios
                    xhat_scenario_names+= refmodel.scenario_names_creator(mult*(nk-nk_m1),
                                                                          start=self.ScenCount)
                    self.ScenCount+= mk-mk_m1
            
            #Computing xhat_k
            xgo = self.xhat_gen_options.copy()
            xgo.pop("solvername", None)  # it will be given explicitly
            xgo.pop("solver_options", None)  # it will be given explicitly
            xgo.pop("scenario_names", None)  # given explicitly
            xhat_k = self.xhat_generator(xhat_scenario_names,
                                         solvername=self.solvername,
                                         solver_options=self.solver_options,
                                         **xgo)
            
            #Computing n_k and associated scenario names
            if self.multistage:
                self.SeedCount += sputils.number_of_nodes(xhat_branching_factors)
                
                gap_branching_factors = ciutils.scalable_branching_factors(lower_bound_k, self.options['branching_factors'])
                nk = np.prod(gap_branching_factors)
                estimator_scenario_names = refmodel.scenario_names_creator(nk)
                sample_options = {'branching_factors':gap_branching_factors, 'seed':self.SeedCount}
            else:
                nk = self.ArRP *int(np.ceil(lower_bound_k/self.ArRP))
                assert nk>= nk_m1, "Our sample size should be increasing"
                if (k%self.kf_Gs==0):
                    #We use only new scenarios to compute gap estimators
                    estimator_scenario_names = refmodel.scenario_names_creator(nk,
                                                                               start=self.ScenCount)
                    self.ScenCount+=nk
                else:
                    #We reuse the previous scenarios
                    estimator_scenario_names+= refmodel.scenario_names_creator((nk-nk_m1),
                                                                               start=self.ScenCount)
                    self.ScenCount+= (nk-nk_m1)
                sample_options = None
            
            
            #Computing G_k and s_k
            self.options['num_scens'] = nk
            estim = ciutils.gap_estimators(xhat_k, self.refmodelname,
                                           solving_type=self.solving_type,
                                           scenario_names=estimator_scenario_names,
                                           sample_options=sample_options,
                                           ArRP=self.ArRP,
                                           options=self.options,
                                           scenario_denouement=scenario_denouement,
                                           solvername=self.solvername,
                                           solver_options=self.solver_options)
            if self.multistage:
                self.SeedCount = estim['seed']
            Gk,sk = estim['G'],estim['s']

            if (k%10==0) and global_rank==0:
                print(f"k={k}")
                print(f"n_k={nk}")
                print(f"G_k={Gk}")
                print(f"s_k={sk}")
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
        global_toc(f"G={Gk} sk={sk}; xhat has been computed with {nk*mult} observations.")
        return {"T":T,"Candidate_solution":final_xhat,"CI":CI,}

if __name__ == "__main__":
    # for developer testing
    solvername = "cplex"
    
    refmodel = "mpisppy.tests.examples.farmer"
    farmer_opt_dict = {"crops_multiplier":3}
    
    # create three options configurations, then use one of them

    # relative width
    optionsBM = config.Config()
    optionsBM.quick_assign('h', float, 0.2)
    optionsBM.quick_assign('hprime', float, 0.015,)
    optionsBM.quick_assign('eps', float, 0.5,)
    optionsBM.quick_assign('epsprime', float, 0.4,)
    optionsBM.quick_assign("p", float, 0.2)
    optionsBM.quick_assign("q", float, 1.2)
    optionsBM.quick_assign("solvername", str, solvername)
    optionsBM.quick_assign("stopping", str, "BM")  # TBD use this and drop stopping_criterion from the constructor
    optionsBM.quick_assign("xhat_gen_options", dict, farmer_opt_dict)

    # fixed width, fully sequential
    optionsFSP = config.Config()
    optionsFSP.quick_assign('eps', float,  50.0)
    optionsFSP.quick_assign('solvername', str,  solvername)
    optionsFSP.quick_assign("c0", int, 50)  # starting sample size)
    optionsFSP.quick_assign("xhat_gen_options", dict, farmer_opt_dict)
    optionsFSP.quick_assign("ArRP", float, 2)  # this must be 1 for any multi-stage problems
    optionsFSP.quick_assign("stopping", str, "BPL")
    optionsFSP.quick_assign("xhat_gen_options", dict, farmer_opt_dict)

    # fixed width sequential with stochastic samples
    optionsSSP = config.Config()
    optionsSSP.quick_assign('eps', float,  1.0)
    optionsSSP.quick_assign('solvername', str,  solvername)
    optionsSSP.quick_assign("n0min", int, 200)   # only for stochastic sampling
    optionsSSP.quick_assign("stopping", str, "BPL")
    optionsSSP.quick_assign("xhat_gen_options", dict, farmer_opt_dict)

    # change the options argument and stopping criterion
    our_pb = SeqSampling(refmodel,
                          xhat_generator_farmer,
                          optionsFSP,
                          stochastic_sampling=False,  # maybe this should move to the options dict?
                          stopping_criterion="BPL",
                          )
    res = our_pb.run()

    print(res)        
