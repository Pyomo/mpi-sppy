import time
import logging
import math

import numpy as np
import mpisppy.MPI as MPI
import csv
import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
import mpisppy.spopt

from mpisppy.utils.prox_approx import ProxApproxManager
from mpisppy import global_toc
from pyomo.repn import generate_standard_repn
class CGBase(mpisppy.spopt.SPOpt):
    """ Base class for all CG-based algorithms.

        Attributes:
            local_scenarios (dict):
                Dictionary mapping scenario names (strings) to scenarios (Pyomo
                concrete model objects). These are only the scenarios managed by
                the current rank (not all scenarios in the entire model).
            comms (dict):
                Dictionary mapping node names (strings) to MPI communicator
                objects.
            local_scenario_names (list):
                List of local scenario names (strings). Should match the keys
                of the local_scenarios dict.
            sp_solver_options (dict): from options
                Dictionary of subproblem solver options provided in options.
            mp_solver_options (dict): from options
                Dictionary of master problem solver options provided in options.    

        Args:
            options (dict):
                Options for the column generation algorithm.
            all_scenario_names (list):
                List of all scenario names in the model (strings).
            scenario_creator (callable):
                Function which takes a scenario name (string) and returns a
                Pyomo Concrete model with required attributes attached.
            scenario_denouement (callable, optional):
                Function which does post-processing and reporting.
            all_nodenames (list, optional):
                List of all node names (strings). Can be `None` for two-stage
                problems.
            mpicomm (MPI comm, optional):
                MPI communicator to use between all scenarios. Default is
                `MPI.COMM_WORLD`.
            scenario_creator_kwargs (dict, optional):
                Keyword arguments passed to `scenario_creator`.
            extensions (object, optional):
                CG extension object.
            extension_kwargs (dict, optional):
                Keyword arguments to pass to the extensions.
            variable_probability (callable, optional):
                Function to set variable specific probabilities.
    """
    


    def __init__(
            self,
            options,
            all_scenario_names,
            scenario_creator,
            scenario_denouement=None,
            all_nodenames=None,
            mpicomm=None,
            scenario_creator_kwargs=None,
            extensions=None,
            extension_kwargs=None,
            variable_probability=None,
        ):
        self._CGIter = 0  
        """ CGBase constructor. """
        super().__init__(
            options,
            all_scenario_names,
            scenario_creator,
            scenario_denouement=scenario_denouement,
            all_nodenames=all_nodenames,
            mpicomm=mpicomm,
            extensions=extensions,
            extension_kwargs=extension_kwargs,
            scenario_creator_kwargs=scenario_creator_kwargs,
            variable_probability=variable_probability,
        )
        global_toc("Initializing CGBase")
       
        # Get indices of nonanticipative variables from any scenario
        any_scen = next(iter(self.local_scenarios.values()))
        self.nonant_indices = list(any_scen._mpisppy_data.nonant_indices.keys())
        # Set up column slots for the master problem
        self.cols = list(range(10*self.options["CGIterLimit"]))#TODO check this
        
        # Track next available column slot for each scenario
        self.next_col = {
            sname: 0 for sname in self.all_scenario_names
        }
        # Track unique columns to avoid duplicates
        self.column_hashes = {sname: set() for sname in self.all_scenario_names}
        
        self.RUB=None
        self.LB_current=None
        self.LB_past=None
        self.best_bound_obj_val=None
        self.mp = None
        self.pi_values=None
        self.mu_values= None
        self.pi_last=None
        self.MIPgap=100
        self.conv=100 
        self.initial_columns_list=None
        self.options_check()


        

    def options_check(self):
        """
        Check whether the options in the `options` attribute are valid for CGBase.

        Required options are:

        - solver_name (string): The name of the solver to use for the master and subproblems.
        - CGIterLimit (int): The maximum number of CG iterations to execute.
        - convthresh (float): The convergence tolerance of the CG algorithm.
        - verbose (boolean): Flag indicating whether to display verbose output.
        - display_progress (boolean): Flag indicating whether to display
          information about the progression of the algorithm.
        - sp_solver_options (dict): Dictionary of solver options to use for solving the subproblems.
        - mp_solver_options (dict): Dictionary of solver options to use for solving the master problem.
        """
        required = ["solver_name", "CGIterLimit","convthresh",
            "verbose", "display_progress","sp_solver_options",
            "mp_solver_options"]
        self._options_check(required, self.options)

    

    def CG_Prep(self):
        """
        Build master problem, compute nonanticipative variables cost coefficients, and set dual parameters.
        """
                
        self.compute_nonant_obj_coefs()
        if(self.cylinder_rank == 0):
            self.mp=self.build_master_model()
        
        self.attach_duals_for_subproblem()
    
    def attach_duals_for_subproblem(self):
        """
        Attach the duals to the models in `local_scenarios` for use when setting the reduced cost.
        """
        for (sname, scenario) in self.local_scenarios.items():
            
            scenario._mpisppy_model.pi = pyo.Param(
                scenario._mpisppy_data.nonant_indices.keys(),
                initialize=0.0,
                mutable=True
            )
            scenario._mpisppy_model.mu = pyo.Param(
                initialize=0.0,
                mutable=True
            )

           
    def build_master_model(self):
        """Build the restricted master problem (RMP)."""
        rmp= pyo.ConcreteModel()
        rmp.nonant_indices = pyo.Set(initialize=self.nonant_indices)
        rmp.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        rmp.scen = pyo.Set(initialize=list(self.all_scenario_names))
        rmp.cols = pyo.Set(initialize=self.cols)  
        rmp.col_cost = pyo.Param(rmp.scen, rmp.cols, initialize=0.0, mutable=True)
        rmp.X_val = pyo.Param(rmp.nonant_indices,rmp.scen, rmp.cols, initialize=0.0, mutable=True)
        rmp.col_is_active = pyo.Param(rmp.scen,rmp.cols, initialize=0, mutable=True)

        rmp.w = pyo.Var(rmp.scen, rmp.cols, initialize=0, domain=pyo.NonNegativeReals)
        rmp.xbar = pyo.Var(rmp.nonant_indices, domain=pyo.Reals)
        

        # Nonanticipativity constraint
        def _nonant_rule(m, r,i, s):
            return sum( m.X_val[r, i, s, c] * m.col_is_active[s, c] * m.w[s, c] for c in m.cols)== m.xbar[r,i]
        def _relaxed_nonant_rule(m, r,i, s):
            return sum( m.X_val[r, i, s, c] * m.col_is_active[s, c] * m.w[s, c] for c in m.cols)<= m.xbar[r,i]
        if self.options["relaxed_nonant"]:
            rmp.NonAnt = pyo.Constraint(rmp.nonant_indices, rmp.scen, rule=_relaxed_nonant_rule)
        else:
            rmp.NonAnt = pyo.Constraint(rmp.nonant_indices, rmp.scen, rule=_nonant_rule)

        # Convexity constraint
        def _convexity_rule(m, s):
            return sum( m.col_is_active[s, c] * m.w[s, c] for c in m.cols) == 1.0
        rmp.Convexity = pyo.Constraint(rmp.scen, rule=_convexity_rule)

        # Objective
        # Assumes master problem is a minimization problem     
        expr = sum(self.nonant_obj_coef[idx] * rmp.xbar[idx] for idx in self.nonant_indices)
        rmp.obj = pyo.Objective(rule=expr+sum(
            rmp.col_cost[s, c] * rmp.col_is_active[s, c] * rmp.w[s, c]
            for s in rmp.scen
            for c in rmp.cols), sense=pyo.minimize)
        
        return rmp
    
    def compute_nonant_obj_coefs(self):
        """
        Build a dictionary of coefficients for nonanticipative variables in the original objective.
        """
        # Use any scenario (they all have the same nonant indices and objective structure)
        sname = next(iter(self.local_scenarios))
        scenario = self.local_scenarios[sname]
        original_obj = self.saved_objectives[sname]
        repn = generate_standard_repn(original_obj, quadratic=False)
        linear_coefs = repn.linear_coefs
        linear_vars  = repn.linear_vars

        coef_dict = {}
        for a, v in zip(linear_coefs, linear_vars):
            vid = id(v)
            if vid in scenario._mpisppy_data.varid_to_nonant_index:
                ndn_i = scenario._mpisppy_data.varid_to_nonant_index[vid]
                coef_dict[ndn_i] = a
        self.nonant_obj_coef = coef_dict    

    def set_subproblem_objective(self):
        """
        Set the reduced-cost objective for the column generation subproblem
        for each scenario.
        """

        for (sname, scenario) in self.local_scenarios.items():
            model = scenario
            original_obj = self.saved_objectives[sname]
            prob=scenario._mpisppy_probability
            model.base_obj_expr = pyo.Expression(expr=prob*original_obj.expr)
            is_min_problem = original_obj.is_minimizing()

            # Remove nonanticipative variable terms from base objective
            repn = generate_standard_repn(model.base_obj_expr, quadratic=False)
            const = repn.constant
            linear_coefs = repn.linear_coefs
            linear_vars  = repn.linear_vars

            expr = const
            for a, v in zip(linear_coefs, linear_vars):
                id_var = id(v)
                if id_var in scenario._mpisppy_data.varid_to_nonant_index.keys():
                    continue
                expr += a * v
            model.base_obj_expr = pyo.Expression(expr=expr)

            # Add dual terms (pi and mu)
            model.NonantDualExpr = pyo.Expression(
                expr=sum(
                    model._mpisppy_model.pi[ndn_i] * xvar
                    for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items()
                )
            )
            model.MuConstExpr = pyo.Expression(expr=model._mpisppy_model.mu)
            redcost_term=model.NonantDualExpr + model.MuConstExpr

            # Deactivate original objective and set reduced-cost objective
            original_obj.deactivate()

            if is_min_problem:
                model.obj = pyo.Objective(
                    expr=model.base_obj_expr - redcost_term,
                    sense=pyo.minimize
                )
            else:
                model.obj = pyo.Objective(
                    expr=model.base_obj_expr + redcost_term,
                    sense=pyo.maximize
                )
       
    def extract_duals_from_mp(self):
        """
        Extract dual values for nonanticipativity and convexity constraints from the master problem.
        """
        pi = {}  
        mu = {}  

        for i in self.mp.nonant_indices:
            for s in self.mp.scen:
                constr = self.mp.NonAnt[i, s]
                pi[s, i] = self.mp.dual[constr]

        for s in self.mp.scen:
            constr = self.mp.Convexity[s]
            mu[s] = self.mp.dual[constr]
        
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

  
        c = self.next_col[sname]

        if c not in self.cols:
            raise RuntimeError(
                f"No free column slots for scenario {sname}. "
                f"Max columns per scenario: {len(self.C)}"
            )

 
        self.mp.col_cost[sname, c] = float(cost)

        for i in self.nonant_indices:
            val = x_vec[i]
            self.mp.X_val[i, sname, c] = round(float(val), 8)

        self.mp.col_is_active[sname, c] = 1


        self.next_col[sname] = c + 1
    
            
    def update_subproblem_duals_from_mp(self, pi_values, mu_values):
        """
        Update scenario-dependent duals after solving the master problem.

        Args:
            pi_values (dict): Dual values for nonanticipativity constraints.
            mu_values (dict): Dual values for convexity constraints.
        """

        if not hasattr(self, "mp"):
            raise RuntimeError("MP model (self.mp) has not been built")
        
        for sname, scenario in self.local_scenarios.items():
            model = scenario
            if pi_values is not None:
                for i in scenario._mpisppy_data.nonant_indices.keys():
                    key = (sname, i)
                    if key not in pi_values:
                        raise KeyError(
                            f"Missing dual value for NonAnt[{i}, {sname}] "
                            f"when updating pi"
                        )
                    model._mpisppy_model.pi[i] = pi_values[key]
            
            if mu_values is not None:
                if sname not in mu_values:
                    raise KeyError(
                        f"Missing dual value for Convexity[{sname}] "
                        f"when updating mu"
                    )
                model._mpisppy_model.mu = mu_values[sname]


    def build_initial_columns(self, initial_xbar=None):
        """
        Build an initial set of columns based on a given value for nonant variables.

        Args:
            initial_xbar (dict or None): Initial values for nonant variables. If None, uses 0.0 for all.
        """

        if initial_xbar is None:
            initial_xbar = {i: 0.0 for i in self.nonant_indices}
        
        for sname, scenario in self.local_scenarios.items():
            for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items():
                target_val = initial_xbar[ndn_i]
                xvar.fix(target_val)
        
        local_results = self.build_columns()
        for sname, scenario in self.local_scenarios.items():
            for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items():
                xvar.unfix()
            
        return local_results
    
    def build_columns(self):
        """
        Build columns for all local scenarios.
        """
        
        self.solve_loop( self.options["sp_solver_options"], tee=False,warmstart=True)
        local_results = []
        for sname, scenario in self.local_scenarios.items():
            sname, red_cost, scen_cost, xvec = self.build_columns_from_subproblem_solutions(sname, scenario)
            local_results.append((sname, red_cost, scen_cost, xvec))
        return local_results


    def build_columns_from_subproblem_solutions(self, sname, scenario):
        """
        Read the solutions of local subproblems and build columns for the MP.
        """
        model = scenario
        scen_cost=None
        red_cost=None
        if hasattr(model, "base_obj_expr"):
            scen_cost = pyo.value(model.base_obj_expr)
        if hasattr(model, "base_obj_expr"):
            red_cost =  model._mpisppy_data.outer_bound
        x_vec = {ndn_i: pyo.value(xvar) for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items()}
        return sname, red_cost, scen_cost,x_vec

    def Iter0(self):
        """
        Create solvers and build initial columns and add them to the master problem.
        """
        
        global_toc("Creating solvers")
        self._create_solvers()
      

        # RMP (master) solver
        master_solver_name = self.options["solver_name"]
        master_solver_opts = self.options["mp_solver_options"]
        self.master_solver = pyo.SolverFactory(master_solver_name)
        self.master_solver.options= master_solver_opts
        #Solve all scenarios separatedly to generate initial columns
        local_results = self.build_columns()
        all_results = self.mpicomm.gather(local_results, root=0)
        if self.cylinder_rank == 0:
            self.initial_columns_list= all_results
        self.initial_columns_list = self.mpicomm.bcast(self.initial_columns_list, root=0)
        
        self.set_subproblem_objective()
        # 2) Build initial columns (one per scenario)
        global_toc("Building initial columns")
         
        # 2) Build initial columns (one per scenario)
        global_toc("Building initial columns")
        local_results=self.build_initial_columns(initial_xbar=None)
        all_results = self.mpicomm.gather(local_results, root=0)
        if self.cylinder_rank == 0:
            self.add_columns_to_mp_from_results(all_results)

        #For creating all columns with the deterministic solutions
        for rank_results in self.initial_columns_list:
            if rank_results is not None:
                # rank_results can be a list of tuples, or a single tuple
                if isinstance(rank_results, list):
                    for result in rank_results:
                        sname, red_cost, scen_cost, xvec = result
                        local_results=self.build_initial_columns(initial_xbar=xvec)
                        all_results = self.mpicomm.gather(local_results, root=0)
                        if self.cylinder_rank == 0:
                            self.add_columns_to_mp_from_results(all_results)

                else:
                    # If only one tuple per rank)
                    sname, red_cost, scen_cost, xvec = rank_results
                    local_results=self.build_initial_columns(initial_xbar=xvec)
                    all_results = self.mpicomm.gather(local_results, root=0)
                    if self.cylinder_rank == 0:
                        self.add_columns_to_mp_from_results(all_results)



    
    def solve_loop(self, solver_options=None,
                   use_scenarios_not_subproblems=False,
                   dtiming=False,
                   gripe=False,
                   disable_pyomo_signal_handling=False,
                   tee=False,
                   verbose=False,
                   need_solution=True,
                   warmstart=sputils.WarmstartStatus.FALSE):
        """ Loop over `local_subproblems` and solve them in a manner
        dicated by the arguments.

        In addition to changing the Var values in the scenarios, this function
        also updates the `_PySP_feas_indictor` to indicate which scenarios were
        feasible/infeasible.

        Args:
            solver_options (dict, optional):
                The scenario solver options.
            use_scenarios_not_subproblems (boolean, optional):
                If True, solves individual scenario problems, not subproblems.
                This distinction matters when using bundling. Default is False.
            dtiming (boolean, optional):
                If True, reports solve timing information. Default is False.
            gripe (boolean, optional):
                If True, output a message when a solve fails. Default is False.
            disable_pyomo_signal_handling (boolean, optional):
                True for asynchronous PH; ignored for persistent solvers.
                Default False.
            tee (boolean, optional):
                If True, displays solver output. Default False.
            verbose (boolean, optional):
                If True, displays verbose output. Default False.
            need_solution (boolean, optional):
                If True, raises an exception if a solution is not available.
                Default True
            warmstart (bool, optional):
                If True, warmstart the subproblem solves. Default False.
        """

        """ Developer notes:

        This function assumes that every scenario already has a
        `_solver_plugin` attached.
        The arguments `use_scenarios_not_subproblems` and `disable_pyomo_signal_handling`
        are not tested in this implementation and should be left as False.

        """
        self.update_subproblem_duals_from_mp(self.pi_values, self.mu_values)
        
        super().solve_loop(
            solver_options,
            use_scenarios_not_subproblems,
            dtiming,
            gripe,
            disable_pyomo_signal_handling,
            tee,
            verbose,
            need_solution,
            warmstart,
        )


    def add_columns_to_mp_from_results(self, all_results):
        """
        Add columns to the master problem from subproblem results.

        Args:
            all_results (list): List of results from all ranks, each containing scenario name,
                reduced cost, scenario cost, and variable values.

        Returns:
            float: The sum of reduced costs
        """
        sum_redcosts=0
        for rank_results in all_results:
            if rank_results is not None:
                if isinstance(rank_results, list):
                    for result in rank_results:
                        sname, red_cost, scen_cost, xvec = result
                        self.add_column_for_scenario(sname, scen_cost, xvec)
                        sum_redcosts+=red_cost
                else:
                    sname, red_cost, scen_cost, xvec = rank_results
                    self.add_column_for_scenario(sname, scen_cost, xvec)
                    sum_redcosts+=red_cost
        return sum_redcosts                

    def _build_columns_from_xhat_list(self, xhat_list):
        """
        Build columns from a list of xhat vectors.

        Args:
            xhat_list (list): List of 1D arrays, each representing a xhat vector with its cost.

        Returns:
            list: List of tuples with scenario name, reduced cost, scenario cost, and variable values.
        """
        a_dict = self.nonant_obj_coef
        local_results = []

        for xhat in xhat_list:
            ci = 0
            for sname, scenario in self.local_scenarios.items():
                xvec = {}
                for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items():
                    xvec[ndn_i] = xhat[ci]
                    ci += 1
                base_scen_cost = scenario._mpisppy_probability * xhat[ci]
                ci += 1
                correction = sum(a_dict.get(ndn_i, 0.0) * xval for ndn_i, xval in xvec.items())
                scen_cost = base_scen_cost - scenario._mpisppy_probability * correction
                reduced_cost = 0
                local_results.append((sname, reduced_cost, scen_cost, xvec))
        return local_results
    
    def build_columns_from_spoke(self, xhat):
        """
        Build columns for each scenario based on a single xhat.

        Args:
            xhat (array-like): Vector of nonant values and scenario costs.

        Returns:
            list: List of tuples with scenario name, reduced cost, scenario cost, and variable values.
        """
        return self._build_columns_from_xhat_list([xhat])

    def build_columns_recent_xhats(self):
        """
        Build columns using all recent xhats stored in self.spcomm.recent_xhats_list.

        Returns:
            list: List of tuples with scenario name, reduced cost, scenario cost, and variable values.
        """
        return self._build_columns_from_xhat_list(self.spcomm.recent_xhats_list)    
    
    def iterk_loop(self):
        """ Perform all CG iterations after iteration 0.

        This function terminates if any of the following occur:

        1. The maximum number of iterations is reached.
        2. The hub tells it to terminate.
        3. A default convergence criteria are met (i.e. the convergence value falls below the
           user-specified threshold).

        Args: None

        """
        verbose = self.options["verbose"]
        dprogress = self.options["display_progress"]
        dtiming = self.options["display_timing"]
        
        max_iterations = int(self.options["CGIterLimit"])
        

        for self._CGIter in range(1, max_iterations+1):
            iteration_start_time = time.time()

            if dprogress:
                global_toc(f"Initiating CG Iteration {self._CGIter}\n", self.cylinder_rank == 0)

            # Solve master problem and extract duals on rank 0
            if self.cylinder_rank == 0:
                rmp_obj=self.solve_master_problem()
                self.RUB=rmp_obj
                self.pi_values, self.mu_values = self.extract_duals_from_mp()
                
            # Broadcast dual values to all ranks
            self.pi_values = self.mpicomm.bcast(self.pi_values, root=0)
            self.mu_values = self.mpicomm.bcast(self.mu_values, root=0)

            teeme = (
                "tee-rank0-solves" in self.options
                 and self.options["tee-rank0-solves"]
                and self.cylinder_rank == 0
            )
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
                
                # Update bounds and convergence metric
                if self.best_bound_obj_val is None:
                    self.best_bound_obj_val = self.LB_current
                else:
                    self.best_bound_obj_val = max(self.LB_current, self.best_bound_obj_val)
                self.conv=(self.RUB-self.best_bound_obj_val)/abs(self.best_bound_obj_val)
                
                if dprogress:
                    print("")
                    print("After CG Iteration",self._CGIter)
                    print("Bounds: ",self.best_bound_obj_val,self.RUB)
                    print("Scaled CG Convergence Metric=",self.conv)
                    print("Iteration time: %6.2f" % (time.time() - iteration_start_time))
                    print("Elapsed time:   %6.2f" % (time.perf_counter() - self.start_time))

            # Broadcast updated bound and convergence metric
            self.best_bound_obj_val = self.mpicomm.bcast(self.best_bound_obj_val, root=0)
            self.conv = self.mpicomm.bcast(self.conv, root=0)
            
            if self.spcomm and self.spcomm.is_converged():
                global_toc("Cylinder convergence", self.cylinder_rank == 0)
                break



        else: # no break, (self._CGHIter == max_iterations)
            # NOTE: If we return for any other reason things are reasonably in-sync.
            #       due to the convergence check. However, here we return we'll be
            #       out-of-sync because of the solve_loop could take vasty different
            #       times on different threads. This can especially mess up finalization.
            #       As a guard, we'll put a barrier here.
            self.mpicomm.Barrier()
            global_toc("Reached user-specified limit=%d on number of CG iterations" % max_iterations, self.cylinder_rank == 0)


    def solve_master_problem(self):
        """
        Solve the current master problem (RMP) with all available columns.
        Returns:
            float: The optimal objective value of the master problem.
        """
        self.master_solver.solve(self.mp, tee=False)
        obj_value = pyo.value(self.mp.obj)
        return obj_value
    
    def solve_ip_master_problem(self):
        """
        Solve the current master problem (RMP) with all available columns.
        Returns:
            float: The optimal objective value of the master problem.
        """
        for (s, c) in self.mp.w.index_set():
            var = self.mp.w[s, c]
            var.domain = pyo.Binary 
        
        self.master_solver.solve(self.mp)
        obj_value = pyo.value(self.mp.obj)
        return obj_value
    
    def post_loops(self, extensions=None):
        dprogress = self.options["display_progress"]
        UB_value=None
        if self.cylinder_rank == 0:
            print("")
            print("Solving integer master problem")
            print("")
            UB_value=self.solve_ip_master_problem()
            print('IP bounds', UB_value, self.best_bound_obj_val)

        return UB_value
       
if __name__ == "__main__":
    print ("No main for CGBase")

