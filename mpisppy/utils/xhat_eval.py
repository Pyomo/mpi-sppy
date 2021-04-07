# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Code to evaluate a given x-hat, but given as a nonant-cache
# To test: python xhat_eval.py --num-scens=3 --EF-solver-name=cplex

import shutil
import pyomo.environ as pyo
import mpisppy.phbase
import mpi4py.MPI as mpi
import mpisppy.utils.sputils as sputils
    
fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

print("WHAT ABOUT MULTI-STAGE")

############################################################################
class Xhat_Eval(mpisppy.phbase.PHBase):
    """ PH. See PHBase for list of args. """

    #======================================================================
    def evaluate(self, nonant_cache):
        """ Compute the expected value.

        Args:
            nonant_cache(numpy vector): special numpy vector with nonant values (see ph)

        Returns:
            Eobj (float or None): Expected value (or None if infeasible)

        """
        verbose = self.PHoptions['verbose']

        self.subproblem_creation(verbose)
        self._create_solvers()
        self._fix_nonants(nonant_cache, verbose=False)

        solver_options = None  # ???
        
        self.solve_loop(solver_options=solver_options,
                        dis_prox=False, # Important
                        use_scenarios_not_subproblems=True,  # ???
                        gripe=True, 
                        tee=False,
                        verbose=verbose)

        Eobj = self.Eobjective(verbose)
        
        return Eobj


if __name__ == "__main__":
    #==============================
    # hardwired by dlw for debugging (this main is like MMW, but strange)
    import mpisppy.tests.examples.farmer as refmodel
    import mpisppy.utils.amalgomator as ama

    # IMPORTANT: we doing MMW out of order!!!!!
    # (because of using canned software we want to test)
    # We get the MMW right term, then xhat, then the MMW left term.
    
    # do the right term of MMW (9) using the first scenarios
    ama_options = {"EF-2stage": True}   # 2stage vs. mstage
    ama_object = ama.from_module("mpisppy.tests.examples.farmer", ama_options)
    ama_object.run()
    print(f"inner bound=", ama_object.best_inner_bound)
    # This the right term of LHS of MMW (9)
    print(f"outer bound=", ama_object.best_outer_bound)

    
    ############### now get an xhat using different scenarios
    # (use use the ama args to get problem parameters)
    ScenCount = ama_object.args.num_scens
    scenario_creator = refmodel.scenario_creator
    scenario_denouement = refmodel.scenario_denouement
    crops_multiplier = ama_object.args.crops_multiplier
    solvername = ama_object.args.EF_solver_name

    scenario_creator_kwargs = {
        "use_integer": False,
        "crops_multiplier": crops_multiplier,
    }

    # different scenarios for xhat
    scenario_names = ['Scenario' + str(i) for i in range(ScenCount, 2*ScenCount)]

    ef = sputils.create_EF(
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    solver = pyo.SolverFactory(solvername)
    if 'persistent' in solvername:
        solver.set_instance(ef, symbolic_solver_labels=True)
        solver.solve(tee=False)
    else:
        solver.solve(ef, tee=False, symbolic_solver_labels=True,)

    print(f"Xhat in-sample objective: {pyo.value(ef.EF_Obj)}")
    

    ########### get the nonants (the xhat)
    # NOTE: we probably should do an assert or two to make sure Vars match
    nonant_cache = sputils.nonant_cache_from_ef(ef)

    # Create the eval object for the left term of the LHS of (9) in MMW
    # but we are back to using the first scenarios
    MMW_scenario_names = ['scen' + str(i) for i in range(ScenCount)]

    # The options need to be re-done (and phase needs to be split up)
    PHopt = {"iter0_solver_options": None,
             "iterk_solver_options": None,
             "solvername": solvername,
             "verbose": False}
    # TBD: set solver options
    ev = Xhat_Eval(PHopt,
                   MMW_scenario_names,
                   scenario_creator,
                   scenario_denouement,
                   do_options_check=False)
    obj_at_xhat = ev.evaluate(nonant_cache)
    print(f"Expected value at xhat={obj_at_xhat}")  # Left term of LHS of (9)



    
