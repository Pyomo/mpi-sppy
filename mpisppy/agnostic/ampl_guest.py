###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# This code sits between the guest model file wrapper and mpi-sppy
# AMPL is the guest language. Started by DLW June 2024
"""
The guest model file (not this file) provides a scenario creator in Python
that attaches to each scenario a scenario probability (or "uniform")
and the following items to populate the guest dict (aka gd):
  name
  conditional probability
  stage number
  nonant var list

(As of June 2024, we are going to be two-stage only...)

The guest model file (which is in Python) also needs to provide:
  scenario_creator_kwargs
  scenario_names_creator
  scenario_denouement

The signature for the scenario creator in the Python model file for an
AMPL guest is the scenario_name, an ampl model file name, and then
keyword args. It is up to the function to instantiate the model
in the guest language and to make sure the data is correct for the
given scenario name and kwargs.

For AMPL, the _nonant_varadata_list should contain objects obtained
from something like the get_variable method of an AMPL model object.

Not concerning indexes: 
To keep it simple and completely generic, we are throwing away a lot
of index information. This will probably slow down instantantion of 
the objective fuction (see attach_PH_to_objective)
See farmer_ample_agnostic.py for an example where indexes are retained.

"""

import mpisppy.utils.sputils as sputils
import pyomo.environ as pyo

from mpisppy import MPI  # for debuggig
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

class AMPL_guest():
    """
    Provide an interface to a model file for an AMPL guest.
    
    Args:
        model_file_name (str): name of Python file that has functions like scenario_creator
        ampl_file_name (str): name of AMPL file that is passed to the model file
    """
    def __init__(self, model_file_name, ampl_file_name):
        self.model_file_name = model_file_name
        self.model_module = sputils.module_name_to_module(model_file_name)
        self.ampl_file_name = ampl_file_name


    def scenario_creator(self, scenario_name, **kwargs):
        """ Wrap the guest (AMPL in this case) scenario creator

        Args:
            scenario_name (str):
                Name of the scenario to construct.
        """
        def _has_ints(s):
            for _,v in s.getVariables():
                if "binary" in str(v) or "integer" in str(v):
                    return True
            return False
        
        s, prob, nonant_vardata_list, obj_fct = self.model_module.scenario_creator(scenario_name,
                                                                     self.ampl_file_name,
                                                                     **kwargs)
        if len(nonant_vardata_list) == 0:
            raise RuntimeError(f"model file {self.model_file_name} has an empty "
                               f" nonant_vardata_list for {scenario_name =}")
        # In general, be sure to process variables in the same order has the guest does (so indexes match)
        nonant_vars = nonant_vardata_list  # typing aid
        def _vname(v):
            return v[1].name().split('[')[0]
        gd = {
            "scenario": s,
            "nonants": {("ROOT",i): v[1] for i,v in enumerate(nonant_vars)},
            "nonant_fixedness": {("ROOT",i): v[1].astatus()=="fixed" for i,v in enumerate(nonant_vars)},
            "nonant_start": {("ROOT",i): v[1].value() for i,v in enumerate(nonant_vars)},
            "nonant_names": {("ROOT",i): (_vname(v), v[0]) for i, v in enumerate(nonant_vars)},
            "probability": prob,
            "obj_fct": obj_fct,
            "sense": pyo.minimize,
            "BFs": None,
            "has_ints": _has_ints(s),
        }
        ##?xxxxx ? create nonant vars and put them on the ampl model,
        ##?xxxxx create constraints to make them equal to the original nonants

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
        # Attach W's and rho to the guest scenario (mutable params).
        gs = scenario._agnostic_dict["scenario"]  # guest scenario handle
        gd = scenario._agnostic_dict
        nonants = gd["nonants"]
        # (there must be some way to create and assign *mutable* params in on call to AMPL)
        gs.eval("param W_on;")
        gs.eval("let W_on := 0;")
        gs.eval("param prox_on;")
        gs.eval("let prox_on := 0;")
        # we are trusting the order to match the nonant indexes
        #create a nonant_indices set in AMPL
        # This should exactly match the nonant_vars.keys()
        gs.eval(f"set nonant_indices := {{0..{len(nonants)}-1}};")
        # TBD xxxxx check for W on the model already
        gs.eval("param W{nonant_indices};")
        # Note: we should probably use set_values instead of let
        gs.eval("let {i in nonant_indices}  W[i] := 0;")
        # start with rho at zero, but update before solve
        # TBD xxxxx check for rho on the model already
        gs.eval("param rho{nonant_indices};")
        gs.eval("let {i in nonant_indices}  rho[i] := 0;")

    def _disable_prox(self, Ag, scenario):
        scenario._agnostic_dict["scenario"].get_parameter("prox_on").set(0)


    def _disable_W(self, Ag, scenario):
        scenario._agnostic_dict["scenario"].get_parameter("W_on").set(0)


    def _reenable_prox(self, Ag, scenario):
        scenario._agnostic_dict["scenario"].get_parameter("prox_on").set(1)


    def _reenable_W(self, Ag, scenario):
        scenario._agnostic_dict["scenario"].get_parameter("W_on").set(1)


    def attach_PH_to_objective(self, Ag, sname, scenario, add_duals, add_prox):
        # TBD: Deal with prox linearization and approximation later,
        # i.e., just do the quadratic version
        # Assume that nonant_indices is on the AMPL model

        # The host has xbars and computes without involving the guest language

        def _vname(i):
            vtuple = gd['nonant_names'][('ROOT',i)]
            return f"{vtuple[0]}" if vtuple[1] == "" else f"{vtuple[0]}['{vtuple[1]}']"
        
        gd = scenario._agnostic_dict
        gs = gd["scenario"]  # guest scenario handle
        gs.eval("param xbars{nonant_indices};")
        obj_fct = gd["obj_fct"]
        objstr = str(obj_fct)
        assert objstr.split (' ')[0] == "minimize", "We currently assume minimization"

        # Dual term (weights W) (This is where indexes are an issue)
        phobjstr = ""
        if add_duals:
            phobjstr += "W_on * ("
            for i in range(len(gd["nonants"])):
                vname = _vname(i)
                phobjstr += f"W[{i}] * {vname} + "
            phobjstr = phobjstr[:-3] + ")"
        # Prox term (quadratic)
        ####### _see copy_nonants_from_host
        if add_prox:
            phobjstr += " + prox_on * ("
            for i in range(len(gd["nonants"])):
                vname = _vname(i)
                phobjstr += f"(rho[{i}]/2.0) * ({vname} * {vname} - 2.0 * xbars[{i}] * {vname} + xbars[{i}]^2) + "
            phobjstr = phobjstr[:-3] + ")"
        objstr = objstr[:-1] + "+ (" + phobjstr + ");"
        objparts = objstr.split()
        objname = objparts[1]  # has the colon, too
        objstr = objstr.replace(f"minimize {objname}", "minimize phobj:")
        obj_fct.drop()
        gs.eval(objstr)
        gs.eval("delete minus_profit;")    
        currentobj = gs.get_current_objective()
        # see _copy_Ws_...  see also the gams version
        WParamDatas = list(gs.get_parameter("W").instances())
        xbarsParamDatas = list(gs.get_parameter("xbars").instances())
        rhoParamDatas = list(gs.get_parameter("rho").instances())
        gd["PH"] = {
            "W": {("ROOT",i): v for i,v in enumerate(WParamDatas)},
            "xbars": {("ROOT",i): v for i,v in enumerate(xbarsParamDatas)},
            "rho": {("ROOT",i): v for i,v in enumerate(rhoParamDatas)},
            "obj": currentobj,
        }


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
        #### start debugging
        if global_rank == 0 and False:
            WParamDatas = list(gs.get_parameter("W").instances())
            print(f" in _solve_one {WParamDatas =} {global_rank =}")
            xbarsParamDatas = list(gs.get_parameter("xbars").instances())
            print(f" in _solve_one {xbarsParamDatas =} {global_rank =}")
            rhoParamDatas = list(gs.get_parameter("rho").instances())
            print(f" in _solve_one {rhoParamDatas =} {global_rank =}")
        #### stop debugging

        solver_name = s._solver_plugin.name
        gs.set_option("solver", solver_name)    
        if 'persistent' in solver_name:
            raise RuntimeError("Persistent solvers are not currently supported in AMPL agnostic.")
        gs.set_option("presolve", 0)

        solver_exception = None
        try:
            gs.solve()
        except Exception as e:
            solver_exception = e

        if gs.solve_result != "solved":
            s._mpisppy_data.scenario_feasible = False
            if gripe:
                print (f"Solve failed for scenario {s.name} on rank {global_rank}")
                print(f"{gs.solve_result =}")
            s._mpisppy_data._obj_from_agnostic = None
            return

        else:
            s._mpisppy_data.scenario_feasible = True

        if solver_exception is not None and need_solution:
            raise solver_exception


        # For AMPL mips, we need to use the gap option to compute bounds
        # https://amplmp.readthedocs.io/rst/features-guide.html
        # As of Aug 2024, this is not tested...
        mipgap = gs.getValue('_mipgap') if gd["has_ints"] else 0
        objobj = gs.get_current_objective()  # different for xhatters
        objval = objobj.value()
        if gd["sense"] == pyo.minimize:
            s._mpisppy_data.outer_bound = objval - mipgap
        else:
            s._mpisppy_data.inner_bound = objval + mipgap

        # copy the nonant x values from gs to s so mpisppy can use them in s
        # in general, we need more checks (see the pyomo agnostic guest example)
        for ndn_i, gxvar in gd["nonants"].items():
            try:   # not sure this is needed
                float(gxvar.value())
            except:  # noqa
                raise RuntimeError(
                    f"Non-anticipative variable {gxvar.name} on scenario {s.name} "
                    "had no value. This usually means this variable "
                    "did not appear in any (active) components, and hence "
                    "was not communicated to the subproblem solver. ")
            if gxvar.astatus() == "pre":
                raise RuntimeError(
                    f"Non-anticipative variable {gxvar.name} on scenario {s.name} "
                    "was presolved out. This usually means this variable "
                    "did not appear in any (active) components, and hence "
                    "was not communicated to the subproblem solver. ")

            s._mpisppy_data.nonant_indices[ndn_i]._value = gxvar.value()

        s._mpisppy_data._obj_from_agnostic = objval


    # local helper
    def _copy_Ws_xbars_rho_from_host(self, s):
        # This is an important function because it allows us to capture whatever the host did
        # print(f"   {s.name =}, {global_rank =}")
        gd = s._agnostic_dict
        gs = gd["scenario"]  # guest scenario handle
        # AMPL params are tuples (index, value), which are immutable
        # ndn_i is a tuple (name, index)
        if hasattr(s._mpisppy_model, "W"):
            Wdict = {ndn_i[1]: pyo.value(v) for ndn_i, v in s._mpisppy_model.W.items()}
            gs.get_parameter("W").set_values(Wdict)
            rhodict = {ndn_i[1]: pyo.value(v) for ndn_i, v in s._mpisppy_model.rho.items()}
            gs.get_parameter("rho").set_values(rhodict)
            xbarsdict = {ndn_i[1]: pyo.value(v) for ndn_i, v in s._mpisppy_model.xbars.items()}
            gs.get_parameter("xbars").set_values(xbarsdict)
        else:
            pass  # presumably an xhatter; we should check, I suppose
        

    # local helper
    def _copy_nonants_from_host(self, s):
        # values and fixedness;
        gd = s._agnostic_dict
        for ndn_i, gxvar in gd["nonants"].items():
            hostVar = s._mpisppy_data.nonant_indices[ndn_i]
            guestVar = gd["nonants"][ndn_i]
            if guestVar.astatus() == "fixed":
                guestVar.unfix()
            if hostVar.is_fixed():
                guestVar.fix(hostVar._value)
            else:
                guestVar.set_value(hostVar._value)


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


