###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################



import mpisppy.MPI as mpi
import mpisppy.cgbase
import pyomo.environ as pyo
from mpisppy import global_toc
import time
# decorator snarfed from stack overflow - allows per-rank profile output file generation.
def profile(filename=None, comm=mpi.COMM_WORLD):
    pass

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()


############################################################################
class DCG(mpisppy.cgbase.CGBase):
    """ DCG. See CGBase for list of args. """

    #======================================================================
    # uncomment the line below to get per-rank profile outputs, which can
    # be examined with snakeviz (or your favorite profile output analyzer)
    #@profile(filename="profile_out")
    def dualcg_main(self, finalize=True):
        """ Execute the Dual stabilized CG algorithm.

        Args:
            finalize (bool, optional, default=True):
                If True, call `DCG.post_loops()`, if False, do not,
                and return None for Eobj

        Returns:
            tuple:
                Tuple containing
                conv (float):
                    The convergence value (not easily interpretable).
                Eobj (float or `None`):
                    If `finalize=True`, this is the expected, weighted
                    objective value with the proximal term included. This value
                    is not directly useful. If `finalize=False`, this value is
                    `None`.

        NOTE:
            You need an xhat finder either in denoument or in an extension.
        """
        verbose = self.options['verbose']
        self.CG_Prep()

        if (verbose):
            print(f'Calling {self.__class__.__name__} Iter0 on global rank {global_rank}')
        self.Iter0()
        if (verbose):
            print(f'Completed {self.__class__.__name__} Iter0 on global rank {global_rank}')
        
        self.iterk_loop()

        if finalize:
            print("Terminating DCG on rank", self.cylinder_rank)
            Eobj = self.spcomm.BestInnerBound
        else:
            Eobj = None

        return self.conv, Eobj
    
    def build_master_model(self):
        """Build the dual of the restricted master problem (dRMP)."""
        m = pyo.ConcreteModel()
        m.nonant_indices = pyo.Set(initialize=self.nonant_indices)
        m.scen = pyo.Set(initialize=list(self.all_scenario_names))
        m.cuts = pyo.ConstraintList() 

         # Dual variables for nonanticipativity constraints
        m.pi = pyo.Var(m.scen, m.nonant_indices, domain=pyo.Reals)
        # Dual variables for convexity constraints
        m.mu = pyo.Var(m.scen, domain=pyo.Reals)

        # Initial Bundle center 
        # It would be ideal to change it to the actual probability of the scenario
        scenario_probs_all = {sname: 1/len(self.all_scenario_names) for sname in self.all_scenario_names
        }

        def _init_bundle_center_pi(m, scen_name, *idx):
            prob = scenario_probs_all[scen_name]             
            coef = self.nonant_obj_coef[idx]          
            return -prob * coef

        m.bundle_center_pi = pyo.Param(
            m.scen,              
            m.nonant_indices,
            initialize=_init_bundle_center_pi,
            mutable=True,
            )
        
        m.epsilon = pyo.Param(initialize=1, mutable=True)

        def pi_sum_rule(m, *idx):
            return sum(m.pi[s, idx] for s in m.scen) == -self.nonant_obj_coef[idx]

        m.pi_sum = pyo.Constraint(m.nonant_indices, rule=pi_sum_rule)
        
        def mu_sum_expr_rule(m):
            return sum(m.mu[s] for s in m.scen)
        m.mu_sum = pyo.Expression(rule=mu_sum_expr_rule)

        # Objective
        # Assumes master problem is a minimization problem
        def obj_rule(m):
            reg_term = m.epsilon * (
            sum((m.pi[s, i] - m.bundle_center_pi[s, i])**2 for s in m.scen for i in m.nonant_indices)
            )   
            return m.mu_sum - reg_term
        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
        return m   
    
    def update_bundle_center(self):
        """
        Update the bundle center with current duals if the lower bound has improved.
        """
        
        if self.LB_current < self.LB_past:
            return  

        m = self.mp  

        for s in m.scen:
            for i in m.nonant_indices:
                m.bundle_center_pi[s, i] = pyo.value(m.pi[s, i])

        self.last_serious_LB = self.LB_current
    
    def extract_duals_from_mp(self):
        """
        Extract dual values for nonanticipativity and convexity constraints from the dual master problem variables.
        """
        pi = {}  
        mu = {}  
        for i in self.mp.nonant_indices:
            for s in self.mp.scen:
                pi[s, i] = pyo.value(self.mp.pi[s, i])
        
        for s in self.mp.scen:
            mu[s] = pyo.value(self.mp.mu[s])

        return pi, mu
    
    def add_column_for_scenario(self, sname, cost, x_vec):
        """
        Add a new column for scenario sname to the RMP.

        Args:
            sname (str): Scenario name.
            cost (float): Column cost for this scenario.
            x_vec (dict): Value of nonanticipative variable X.
        """
        if sname not in self.all_scenario_names:
            raise KeyError(f"Unknown scenario: {sname}")

        # Check for duplicate column
        key = tuple(round(float(x_vec[i]), 8) for i in self.nonant_indices)
        if key in self.column_hashes[sname]:
            return False

        self.column_hashes[sname].add(key)

        expr = self.mp.mu[sname] + sum(x_vec[i] * self.mp.pi[sname, i] for i in self.nonant_indices)
        self.mp.cuts.add(expr <= cost)

    def iterk_loop(self):
        """ Perform all CG iterations after iteration 0.

        This function terminates if any of the following occur:

        1. The maximum number of iterations is reached.
        2. The hub tells it to terminate.
        3. A default convergence criteria are met (i.e. the convergence value falls below the
           user-specified threshold).

        Args: None

        """
        dprogress = self.options["display_progress"]
        
        max_iterations = int(self.options["CGIterLimit"])
        

        for self._CGIter in range(1, max_iterations+1):
            iteration_start_time = time.time()

            if dprogress:
                global_toc(f"Initiating DCG Iteration {self._CGIter}\n", self.cylinder_rank == 0)

            # Solve master problem and extract duals on rank 0
            if self.cylinder_rank == 0:
                self.solve_master_problem()
                self.RUB=pyo.value(self.mp.mu_sum) 
                self.pi_values, self.mu_values = self.extract_duals_from_mp()
                
            # Broadcast dual values to all ranks
            self.pi_values = self.mpicomm.bcast(self.pi_values, root=0)
            self.mu_values = self.mpicomm.bcast(self.mu_values, root=0)

            # Build columns for all local scenarios
            local_results = self.build_columns()
            all_results = self.mpicomm.gather(local_results, root=0)
            local_results_xhat_recent=None 
            local_results_xfeas=None
            if hasattr(self.spcomm, "sync"):
                self.spcomm.sync()
                if self.spcomm.is_new_latest_xhat:
                    local_results_xhat_recent=self.build_columns_recent_xhats()
                if self.spcomm.is_new_xfeas:
                    local_results_xfeas=self.build_columns_from_spoke(self.spcomm.xfeas)
            all_results_xhat_recent = self.mpicomm.gather(local_results_xhat_recent, root=0)
            all_results_xfeas = self.mpicomm.gather(local_results_xfeas, root=0)

            if self.cylinder_rank == 0:
                sum_redcosts=self.add_columns_to_mp_from_results(all_results)
                self.add_columns_to_mp_from_results(all_results_xhat_recent) 
                self.add_columns_to_mp_from_results(all_results_xfeas) 
                self.LB_current = self.RUB + sum_redcosts
                if self.LB_past is None:
                    self.LB_past = self.LB_current
                
                self.update_bundle_center()
                
                # Update bounds and convergence metric
                if self.best_bound_obj_val is None:
                    self.best_bound_obj_val = self.LB_current
                else:
                    self.best_bound_obj_val = max(self.LB_current, self.best_bound_obj_val)
                
                self.conv=(self.RUB-self.best_bound_obj_val)/abs(self.best_bound_obj_val)
                
                if dprogress:
                    print("")
                    print("After DCG Iteration",self._CGIter)
                    print("Bounds: ",self.best_bound_obj_val,self.RUB)
                    print("Scaled DCG Convergence Metric=",self.conv)
                    print("Iteration time: %6.2f" % (time.time() - iteration_start_time))
                    print("Elapsed time:   %6.2f" % (time.perf_counter() - self.start_time))
                
            self.best_bound_obj_val = self.mpicomm.bcast(self.best_bound_obj_val, root=0)
            self.conv = self.mpicomm.bcast(self.conv, root=0)
            if self.spcomm and self.spcomm.is_converged():
                global_toc("Cylinder convergence", self.cylinder_rank == 0)
                break



        else: # no break, (self._CGIter == max_iterations)
            # NOTE: If we return for any other reason things are reasonably in-sync.
            #       due to the convergence check. However, here we return we'll be
            #       out-of-sync because of the solve_loop could take vasty different
            #       times on different threads. This can especially mess up finalization.
            #       As a guard, we'll put a barrier here.
            self.mpicomm.Barrier()
            global_toc("Reached user-specified limit=%d on number of DCG iterations" % max_iterations, self.cylinder_rank == 0)        


if __name__ == "__main__":
    #==============================
    # hardwired by dlw for debugging
    import mpisppy.tests.examples.farmer as refmodel
    import mpisppy.utils.sputils as sputils

    DCGopt = {}

    DCGopt["solver_name"] = "xpress"
    DCGopt["CGIterLimit"] = 10
    DCGopt["convthresh"] = 0.001
    DCGopt["verbose"] = True
    DCGopt["display_timing"] = True
    DCGopt["display_progress"] = True
    # one way to set up options (never mind that this is not a MIP)
    DCGopt["iter0_solver_options"]\
        = sputils.option_string_to_dict("mipgap=0.01")
    # another way
    DCGopt["iterk_solver_options"] = {"mipgap": 0.001}

    ScenCount = 50
    all_scenario_names = ['scen' + str(i) for i in range(ScenCount)]
    # end hardwire

    scenario_creator = refmodel.scenario_creator
    scenario_denouement = refmodel.scenario_denouement

    dcg = DCG(DCGopt, all_scenario_names, scenario_creator, scenario_denouement)
    dcg.options["CGIterLimit"] = 10
    conv, obj, bnd = dcg.dualcg_main()


    if global_rank == 0:
        print ("E[obj] for converged solution",
               obj)

    dopts = sputils.option_string_to_dict("mipgap=0.001")
