###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# This code sits between the guest model file and mpi-sppy
# Pyomo is the guest language. Started by DLW April 2024
"""
For other guest languages, the corresponding module is
still written in Python, it just needs to interact
with the guest language

The guest model file (not this file) provides a scenario creator in the guest
language that attaches to each scenario a scenario probability (or "uniform")
and the following items to populate the guest dict (aka gd):
  name
  conditional probability
  stage number
  nonant var list

The guest model file also needs to somehow (it might depend on the language)
provide hooks to:
  scenario_creator_kwargs
  scenario_names_creator
  scenario_denouement

  Note: we already have a lot of two-stage models in Pyomo that would
  be handy for testing. All that needs to be done, is to attach
  the nonant varlist as _nonant_vars to the scenario when it is created.
"""
import mpisppy.utils.sputils as sputils
import pyomo.environ as pyo
from pyomo.opt import SolutionStatus, TerminationCondition

from mpisppy import MPI  # for debugging
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

class Pyomo_guest():
    """
    Provide an interface to a model file for a Pyomo guest
    """
    def __init__(self, model_file_name):
        self.model_module = sputils.module_name_to_module(model_file_name)


    def scenario_creator(self, scenario_name, **kwargs):
        """ Wrap the guest (Pyomo in this case) scenario creator

        Args:
            scenario_name (str):
                Name of the scenario to construct.
        """
        s = self.model_module.scenario_creator(scenario_name, **kwargs)
        ### TBD: assert that this is minimization?
        if hasattr(s, "_nonant_vardata_list"): 
            nonant_vars = s._nonant_vardata_list  # a list of vars
        elif hasattr(s, "_mpisppy_node_list"):
            assert len(s._mpisppy_node_list) == 1, "multi-stage agnostic with Pyomo as guest not yet supported."
            nonant_vars = s._mpisppy_node_list[0].nonant_vardata_list
        else:
            raise RuntimeError("Scenario must have either _mpisppy_node_list or _nonant_vardata_list")
        # In general, be sure to process variables in the same order has the guest does (so indexes match)
        gd = {
            "scenario": s,
            "nonants": {("ROOT",i): v for i,v in enumerate(nonant_vars)},
            "nonant_fixedness": {("ROOT",i): v.is_fixed() for i,v in enumerate(nonant_vars)},
            "nonant_start": {("ROOT",i): v._value for i,v in enumerate(nonant_vars)},
            "nonant_names": {("ROOT",i): v.name for i, v in enumerate(nonant_vars)},
            "probability": s._mpisppy_probability,
            "sense": pyo.minimize,
            "BFs": None
            }
        # we don't need to attach nonants to s; the agnostic class does it
        return gd

    #=========
    def scenario_names_creator(self, num_scens,start=None):
        return self.model_module.scenario_names_creator(num_scens,start)


    #=========
    def inparser_adder(self, cfg):
        self.model_module.inparser_adder(cfg)


    #=========
    def kw_creator(self, cfg):
        # creates keywords for scenario creator
        return self.model_module.kw_creator(cfg)

    # This is not needed for PH
    def sample_tree_scen_creator(self, sname, stage, sample_branching_factors, seed,
                                 given_scenario=None, **scenario_creator_kwargs):
        return self.model_module.sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                                               given_scenario, **scenario_creator_kwargs)

    #============================
    def scenario_denouement(self, rank, scenario_name, scenario):
        pass
        # (the fct in farmer won't work because the Var names don't match)
        #self.model_module.scenario_denouement(rank, scenario_name, scenario)


    ############################################################################
    # begin callouts
    # NOTE: the callouts all take the Ag object as their first argument, mainly to see cfg if needed
    # the function names correspond to function names in mpisppy

    def attach_Ws_and_prox(self, Ag, sname, scenario):
        # Attach W's and prox to the guest scenario.
        # Use the nonant index as the index set
        gs = scenario._agnostic_dict["scenario"]  # guest scenario handle
        nonant_idx = list(scenario._agnostic_dict["nonants"].keys())
        assert not hasattr(gs, "W")
        gs.W = pyo.Param(nonant_idx, initialize=0.0, mutable=True)
        assert not hasattr(gs, "W_on")
        gs.W_on = pyo.Param(initialize=0, mutable=True, within=pyo.Binary)
        assert not hasattr(gs, "prox_on")
        gs.prox_on = pyo.Param(initialize=0, mutable=True, within=pyo.Binary)
        assert not hasattr(gs, "rho")
        gs.rho = pyo.Param(nonant_idx, mutable=True, default=Ag.cfg.default_rho)


    def _disable_prox(self, Ag, scenario):
        scenario._agnostic_dict["scenario"].prox_on._value = 0


    def _disable_W(self, Ag, scenario):
        scenario._agnostic_dict["scenario"].W_on._value = 0


    def _reenable_prox(self, Ag, scenario):
        scenario._agnostic_dict["scenario"].prox_on._value = 1


    def _reenable_W(self, Ag, scenario):
        scenario._agnostic_dict["scenario"].W_on._value = 1


    def attach_PH_to_objective(self, Ag, sname, scenario, add_duals, add_prox):
        # TBD: Deal with prox linearization and approximation later,
        # i.e., just do the quadratic version

        ### The host has xbars and computes without involving the guest language
        ### xbars = scenario._mpisppy_model.xbars
        ### but instead, we are going to make guest xbars like other guests


        gd = scenario._agnostic_dict
        gs = gd["scenario"]  # guest scenario handle
        nonant_idx = list(gd["nonants"].keys())
        # for Pyomo, we can just ask what is the active objective function
        # (from some guests, maybe we will have to put the obj function on gd
        objfct = sputils.find_active_objective(gs)
        ph_term = 0
        gs.xbars = pyo.Param(nonant_idx, mutable=True)
        # Dual term (weights W)
        if add_duals:
            gs.WExpr = pyo.Expression(expr= sum(gs.W[ndn_i] * xvar for ndn_i,xvar in gd["nonants"].items()))
            ph_term += gs.W_on * gs.WExpr

            # Prox term (quadratic)
            if (add_prox):
                prox_expr = 0.
                for ndn_i, xvar in gd["nonants"].items():
                    # expand (x - xbar)**2 to (x**2 - 2*xbar*x + xbar**2)
                    # x**2 is the only qradratic term, which might be
                    # dealt with differently depending on user-set options
                    if xvar.is_binary():
                        xvarsqrd = xvar
                    else:
                        xvarsqrd = xvar**2
                    prox_expr += (gs.rho[ndn_i] / 2.0) * \
                        (xvarsqrd - 2.0 * gs.xbars[ndn_i] * xvar + gs.xbars[ndn_i]**2)
                gs.ProxExpr = pyo.Expression(expr=prox_expr)
                ph_term += gs.prox_on * gs.ProxExpr

                if gd["sense"] == pyo.minimize:
                    objfct.expr += ph_term
                elif gd["sense"] == pyo.maximize:
                    objfct.expr -= ph_term
                else:
                    raise RuntimeError(f"Unknown sense {gd['sense'] =}")


    def solve_one(self, Ag, s, solve_keyword_args, gripe, tee=False, need_solution=True):
        # This needs to attach stuff to s (see solve_one in spopt.py)
        # Solve the guest language version, then copy values to the host scenario

        # This function needs to update W on the guest right before the solve

        # We need to operate on the guest scenario, not s; however, attach things to s (the host scenario)
        # and copy to s. If you are working on a new guest, you should not have to edit the s side of things

        # To acommdate the solve_one call from xhat_eval.py, we need to attach the obj fct value to s

        self._copy_Ws_xbars_rho_from_host(s)
        gd = s._agnostic_dict
        gs = gd["scenario"]  # guest scenario handle

        # print(f" in _solve_one  {global_rank =}")
        if global_rank == 0:
            #print(f"{gs.W.pprint() =}")
            #print(f"{gs.xbars.pprint() =}")
            pass
        solver_name = s._solver_plugin.name
        solver = pyo.SolverFactory(solver_name)
        if 'persistent' in solver_name:
            raise RuntimeError("Persistent solvers are not currently supported in the pyomo agnostic example.")
            ###solver.set_instance(ef, symbolic_solver_labels=True)
            ###solver.solve(tee=True)
        else:
            solver_exception = None
            try:
                results = solver.solve(gs, tee=tee, symbolic_solver_labels=True,load_solutions=False)
            except Exception as e:
                results = None
                solver_exception = e

        if (results is None) or (len(results.solution) == 0) or \
                (results.solution(0).status == SolutionStatus.infeasible) or \
                (results.solver.termination_condition == TerminationCondition.infeasible) or \
                (results.solver.termination_condition == TerminationCondition.infeasibleOrUnbounded) or \
                (results.solver.termination_condition == TerminationCondition.unbounded):

            s._mpisppy_data.scenario_feasible = False

            if gripe:
                print (f"Solve failed for scenario {s.name} on rank {global_rank}")
                if results is not None:
                    print ("status=", results.solver.status)
                    print ("TerminationCondition=",
                           results.solver.termination_condition)

            if solver_exception is not None and need_solution:
                raise solver_exception

        else:
            s._mpisppy_data.scenario_feasible = True
            if gd["sense"] == pyo.minimize:
                s._mpisppy_data.outer_bound = results.Problem[0].Lower_bound
            else:
                s._mpisppy_data.outer_bound = results.Problem[0].Upper_bound
            gs.solutions.load_from(results)
            # copy the nonant x values from gs to s so mpisppy can use them in s
            for ndn_i, gxvar in gd["nonants"].items():
                # courtesy check for staleness on the guest side before the copy
                if not gxvar.fixed and gxvar.stale:
                    try:
                        float(pyo.value(gxvar))
                    except:  # noqa
                        raise RuntimeError(
                            f"Non-anticipative variable {gxvar.name} on scenario {s.name} "
                            "reported as stale. This usually means this variable "
                            "did not appear in any (active) components, and hence "
                            "was not communicated to the subproblem solver. ")

                s._mpisppy_data.nonant_indices[ndn_i]._value = gxvar._value

            # the next line ignore bundles (other than proper bundles)
            s._mpisppy_data._obj_from_agnostic = pyo.value(sputils.get_objs(gs)[0])


    # local helper
    def _copy_Ws_xbars_rho_from_host(self, s):
        # This is an important function because it allows us to capture whatever the host did
        # print(f"   {s.name =}, {global_rank =}")
        gd = s._agnostic_dict
        gs = gd["scenario"]  # guest scenario handle
        for ndn_i, gxvar in gd["nonants"].items():
            assert hasattr(s, "_mpisppy_model"),\
                f"what the heck!! no _mpisppy_model {s.name =} {global_rank =}"
            if hasattr(s._mpisppy_model, "W"):
                gs.W[ndn_i] = pyo.value(s._mpisppy_model.W[ndn_i])
                gs.rho[ndn_i] = pyo.value(s._mpisppy_model.rho[ndn_i])
                gs.xbars[ndn_i] = pyo.value(s._mpisppy_model.xbars[ndn_i])
            else:
                # presumably an xhatter
                pass


    # local helper
    def _copy_nonants_from_host(self, s):
        # values and fixedness; 
        gd = s._agnostic_dict
        for ndn_i, gxvar in gd["nonants"].items():
            hostVar = s._mpisppy_data.nonant_indices[ndn_i]
            guestVar = gd["nonants"][ndn_i]
            if guestVar.is_fixed():
                guestVar.fixed = False
            if hostVar.is_fixed():
                guestVar.fix(hostVar._value)
            else:
                guestVar._value = hostVar._value


    def _restore_nonants(self, Ag, s):
        # the host has already restored
        self._copy_nonants_from_host(s)


    def _restore_original_fixedness(self, Ag, s):
        self._copy_nonants_from_host(s)


    def _fix_nonants(self, Ag, s):
        # We are assuming the host did the fixing
        self._copy_nonants_from_host(s)


    def _fix_root_nonants(self, Ag, s):
        # We are assuming the host did the fixing
        self._copy_nonants_from_host(s)


