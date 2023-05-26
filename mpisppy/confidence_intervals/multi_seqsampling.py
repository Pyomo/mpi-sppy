# Copyright 2021 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Code that is producing an xhat and a confidence interval using sequantial sampling 
# This extension of SeqSampling works for multistage, using independent 
# scenarios instead of a single scenario tree.

import pyomo.environ as pyo
import pyomo.common.config as pyofig
import mpisppy.MPI as mpi
import mpisppy.utils.sputils as sputils
import mpisppy.confidence_intervals.confidence_config as confidence_config
from mpisppy.utils import config
import numpy as np
import scipy.stats
import importlib
from mpisppy import global_toc
    
fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

import mpisppy.utils.amalgamator as amalgamator
import mpisppy.utils.xhat_eval as xhat_eval
import mpisppy.confidence_intervals.ciutils as ciutils
from mpisppy.confidence_intervals.seqsampling import SeqSampling
from mpisppy.tests.examples.aircond import xhat_generator_aircond
import mpisppy.confidence_intervals.sample_tree as sample_tree
import mpisppy.confidence_intervals.ciutils as ciutils

class IndepScens_SeqSampling(SeqSampling):
    def __init__(self,
                 refmodel,
                 xhat_generator,
                 cfg,
                 stopping_criterion = "BM",
                 stochastic_sampling = False,
                 solving_type="EF-mstage",
                 ):
        super().__init__(
                 refmodel,
                 xhat_generator,
                 cfg,
                 stochastic_sampling = stochastic_sampling,
                 stopping_criterion = stopping_criterion,
                 solving_type = solving_type)
        self.numstages = len(self.cfg['branching_factors'])+1
        self.batch_branching_factors = [1]*(self.numstages-1)
        self.batch_size = 1
    
    #TODO: Add an override specifier if it exists
    def run(self, maxit=200):
        # Do everything as specified by the options (maxit is provided as a safety net).
        refmodel = self.refmodel
        mult = self.sample_size_ratio # used to set m_k= mult*n_k
        scenario_denouement = refmodel.scenario_denouement if hasattr(refmodel, "scenario_denouement") else None
         #----------------------------Step 0 -------------------------------------#
        #Initialization
        k=1
        
        if self.stopping_criterion == "BM":
            #Finding a constant used to compute nk
            r = 2 #TODO : we could add flexibility here
            j = np.arange(1,1000)
            if self.BM_q is None:
                s = sum(np.power(j,-self.BM_p*np.log(j)))
            else:
                if self.BM_q<1:
                    raise RuntimeError("Parameter BM_q should be greater than 1.")
                s = sum(np.exp(-self.BM_p*np.power(j,2*self.BM_q/r)))
            self.c = max(1,2*np.log(s/(np.sqrt(2*np.pi)*(1-self.confidence_level))))
        
        lower_bound_k = self.sample_size(k, None, None, None)
        
        #Computing xhat_1.
        
        #We use sample_size_ratio*n_k observations to compute xhat_k
        xhat_branching_factors = ciutils.scalable_branching_factors(mult*lower_bound_k, self.cfg['branching_factors'])
        mk = np.prod(xhat_branching_factors)
        self.xhat_gen_kwargs['start_seed'] = self.SeedCount #TODO: Maybe find a better way to manage seed
        xhat_scenario_names = refmodel.scenario_names_creator(mk)

        xgo = self.xhat_gen_kwargs.copy()
        xgo["solver_name"] = self.solver_name
        xgo.pop("solver_options", None)  # it will be given explicitly
        xgo.pop("scenario_names", None)  # it will be given explicitly
        xgo["branching_factors"] = xhat_branching_factors
        xhat_k = self.xhat_generator(xhat_scenario_names,
                                   solver_options=self.solver_options,
                                   **xgo)
        self.SeedCount += sputils.number_of_nodes(xhat_branching_factors)

        #----------------------------Step 1 -------------------------------------#
        #Computing n_1 and associated scenario names
        
        nk = np.prod(ciutils.scalable_branching_factors(lower_bound_k, self.cfg['branching_factors'])) #To ensure the same growth that in the one-tree seqsampling
        estimator_scenario_names = refmodel.scenario_names_creator(nk)
        
        #Computing G_nk and s_k associated with xhat_1
        
        Gk, sk = self._gap_estimators_with_independent_scenarios(xhat_k,
                                                                nk,
                                                                estimator_scenario_names,
                                                                scenario_denouement)

        
        #----------------------------Step 2 -------------------------------------#

        while( self.stop_criterion(Gk,sk,nk) and k<maxit):
        #----------------------------Step 3 -------------------------------------#       
            k+=1
            nk_m1 = nk #n_{k-1}
            mk_m1 = mk
            lower_bound_k = self.sample_size(k, Gk, sk, nk_m1)
            
            #Computing m_k and associated scenario names
            xhat_branching_factors = ciutils.scalable_branching_factors(mult*lower_bound_k, self.cfg['branching_factors'])
            mk = np.prod(xhat_branching_factors)
            self.xhat_gen_kwargs['start_seed'] = self.SeedCount #TODO: Maybe find a better way to manage seed
            xhat_scenario_names = refmodel.scenario_names_creator(mk)
            
            #Computing xhat_k
           
            xgo = self.xhat_gen_kwargs.copy()
            xgo.pop("solver_options", None)  # it will be given explicitly
            xgo.pop("scenario_names", None)  # it will be given explicitly
            xhat_k = self.xhat_generator(xhat_scenario_names,
                                        solver_options=self.solver_options,
                                         **xgo)
            
            #Computing n_k and associated scenario names
            self.SeedCount += sputils.number_of_nodes(xhat_branching_factors)
            
            nk = np.prod(ciutils.scalable_branching_factors(lower_bound_k, self.cfg['branching_factors'])) #To ensure the same growth that in the one-tree seqsampling
            nk += self.batch_size - nk%self.batch_size
            estimator_scenario_names = refmodel.scenario_names_creator(nk)
            
            
            Gk, sk = self._gap_estimators_with_independent_scenarios(xhat_k,nk,estimator_scenario_names,scenario_denouement)

            if (k%10==0):
                print(f"k={k}")
                print(f"n_k={nk}")

        #----------------------------Step 4 -------------------------------------#
        if (k==maxit) :
            raise RuntimeError(f"The loop terminated after {maxit} iteration with no acceptable solution")
        T = k
        final_xhat=xhat_k
        if self.stopping_criterion == "BM":
            upper_bound=self.BM_h*sk+self.BM_eps
        elif self.stopping_criterion == "BPL":
            upper_bound = self.BPL_eps
        else:
            raise RuntimeError("Only BM and BPL criterion are supported yet.")
        CI=[0,upper_bound]
        global_toc(f"G={Gk}")
        global_toc(f"s={sk}")
        global_toc(f"xhat has been computed with {nk*mult} observations.")
        return {"T":T,"Candidate_solution":final_xhat,"CI":CI,}

    """
    def _independent_scenario_creator(self, sname, **scenario_creator_kwargs):
                bfs = [1]*(self.numstages-1)
                snum = sputils.extract_num(sname)
                scenario_creator = self.refmodel.scenario_creator
                return scenario_creator(sname,start_seed=self.SeedCount+snum*(self.numstages-1),**scenario_creator_kwargs)
    """

    def _kw_creator_without_seed(self, cfg):
        kwargs = self.refmodel.kw_creator(cfg)
        kwargs.pop("start_seed")
        return kwargs


    def _gap_estimators_with_independent_scenarios(self, xhat_k, nk,
                                                  estimator_scenario_names, scenario_denouement):
        """ Sample a scenario tree: this is a subtree, but starting from stage 1.
        Args:
            xhat_k (dict[nodename] of list): the solution to lead the walk
            nk (int): number of scenarios,
            estimator_scenario_names(list of str): scenario names
            scenario_denouement (fct): called for each scenario at the end
                 (TBD: drop this arg and just use the function in refmodel)
        Returns:
            Gk, Sk (float): mean and standard devation of the gap estimate
        Note:
            Seed management is mainly in the form of updates to SeedCount

        """
        cfg = config.Config()
        cfg.quick_assign("EF_mstage", bool, True)
        cfg.quick_assign("EF_solver_name", str, self.solver_name)
        cfg.quick_assign("EF_solver_options", dict, self.solver_options)
        cfg.quick_assign("num_scens", int, nk)
        cfg.quick_assign("_mpisppy_probability", float, 1/nk)
        cfg.quick_assign("start_seed", int, self.SeedCount)        
        
        pseudo_branching_factors = [nk]+[1]*(self.numstages-2)
        cfg.quick_assign('branching_factors', pyofig.ListOf(int), pseudo_branching_factors)
        ama = amalgamator.Amalgamator(cfg, 
                                      scenario_names=estimator_scenario_names,
                                      scenario_creator=self.refmodel.scenario_creator,
                                      kw_creator=self.refmodel.kw_creator,
                                      scenario_denouement=scenario_denouement)
        ama.run()
        #Optimal solution of the approximate problem
        zstar = ama.best_outer_bound
        #Associated policies
        xstars = sputils.nonant_cache_from_ef(ama.ef)
        scenario_creator_kwargs = ama.kwargs
        # Find feasible policies (i.e. xhats) for every non-leaf nodes
        local_scenarios = {sname:getattr(ama.ef,sname) for sname in ama.ef._ef_scenario_names}
        xhats,start = sample_tree.walking_tree_xhats(self.refmodelname,
                                                    local_scenarios,
                                                    xhat_k['ROOT'],
                                                    self.cfg['branching_factors'],  # psuedo
                                                    self.SeedCount,
                                                     self.cfg,
                                                    solver_name=self.solver_name,
                                                    solver_options=self.solver_options)
        
        #Compute then the average function value with this policy
        all_nodenames = sputils.create_nodenames_from_branching_factors(pseudo_branching_factors)
        xhat_eval_options = {"iter0_solver_options": None,
                         "iterk_solver_options": None,
                         "display_timing": False,
                         "solver_name": self.solver_name,
                         "verbose": False,
                         "solver_options":self.solver_options}
        ev = xhat_eval.Xhat_Eval(xhat_eval_options,
                                estimator_scenario_names,
                                self.refmodel.scenario_creator,
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
        if global_rank==0:
            print(f"G = {G}")
        sample_var = (ssq - G**2)/(1-prob_sqnorm) #Unbiased sample variance
        sk = np.sqrt(sample_var)
        
        use_relative_error = (np.abs(zstar)>1)
        Gk = ciutils.correcting_numeric(G,cfg,objfct=obj_at_xhat,
                               relative_error=use_relative_error)
        
        self.SeedCount = start
        
        return Gk,sk
    

if __name__ == "__main__":
    # For use by confidence interval develepors who need a quick test.
    solver_name = "cplex"
    #An example of sequential sampling for the aircond model
    import mpisppy.tests.examples.aircond
    bfs = [3,3,2]
    num_scens = np.prod(bfs)
    scenario_names = mpisppy.tests.examples.aircond.scenario_names_creator(num_scens)
    xhat_gen_kwargs = {"scenario_names": scenario_names,
                        "solver_name": solver_name,
                        "solver_options": None,
                        "branching_factors": bfs,
                        "mu_dev": 0,
                        "sigma_dev": 40,
                        "start_ups": False,
                        "start_seed": 0,
                        }

    # create two configs, then use one of them
    optionsBM = config.Config()
    confidence_config.confidence_config(optionsBM)
    confidence_config.sequential_config(optionsBM)
    optionsBM.quick_assign('BM_h', float, 0.55)
    optionsBM.quick_assign('BM_hprime', float, 0.5,)
    optionsBM.quick_assign('BM_eps', float, 0.5,)
    optionsBM.quick_assign('BM_eps_prime', float, 0.4,)
    optionsBM.quick_assign("BM_p", float, 0.2)
    optionsBM.quick_assign("BM_q", float, 1.2)
    optionsBM.quick_assign("solver_name", str, solver_name)
    optionsBM.quick_assign("solver_name", str, solver_name)
    optionsBM.quick_assign("stopping", str, "BM")  # TBD use this and drop stopping_criterion from the constructor
    optionsBM.quick_assign("xhat_gen_kwargs", dict, xhat_gen_kwargs)
    optionsBM.quick_assign("solving_type", str, "EF_mstage")
    optionsBM.add_and_assign("branching_factors", description="branching factors", domain=pyofig.ListOf(int), default=None, value=bfs)
    optionsBM.quick_assign("start_ups", bool, False)
    optionsBM.quick_assign("EF_mstage", bool, True)
    ##optionsBM.quick_assign("cylinders", pyofig.ListOf(str), ['ph'])
       
    # fixed width, fully sequential
    optionsFSP = config.Config()
    confidence_config.confidence_config(optionsFSP)
    confidence_config.sequential_config(optionsFSP)
    optionsFSP.quick_assign('BPL_eps', float,  15.0)
    optionsFSP.quick_assign("BPL_c0", int, 50)  # starting sample size)
    optionsFSP.quick_assign("xhat_gen_kwargs", dict, xhat_gen_kwargs)
    optionsFSP.quick_assign("ArRP", int, 2)  # this must be 1 for any multi-stage problems
    optionsFSP.quick_assign("stopping", str, "BPL")
    optionsFSP.quick_assign("solving_type", str, "EF_mstage")
    optionsFSP.quick_assign("solver_name", str, solver_name)
    optionsFSP.quick_assign("solver_name", str, solver_name)
    optionsFSP.add_and_assign("branching_factors", description="branching factors", domain=pyofig.ListOf(int), default=None, value=bfs)
    optionsFSP.quick_assign("start_ups", bool, False)
    optionsBM.quick_assign("EF_mstage", bool, True)
    ##optionsFSP.quick_assign("cylinders", pyofig.ListOf(str), ['ph'])
   
    aircondpb = IndepScens_SeqSampling("mpisppy.tests.examples.aircond",
                                       xhat_generator_aircond, 
                                       optionsBM,
                                       stopping_criterion="BM"
                                       )

    res = aircondpb.run(maxit=50)
    print(res)

    
    
