# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# JPW and DLW; July 2019; ccopf create scenario instances for line outages
# extended Fall 2019 by DLW
import egret
import egret.models.acopf as eac
import egret.models.ac_relaxations as eac_relax
from egret.data.model_data import ModelData
from egret.parsers.matpower_parser import create_ModelData
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils

import ACtree as ET

import os
import sys
import copy
import scipy
import socket
import numpy as np
import datetime as dt
import mpisppy.MPI as mpi

import pyomo.environ as pyo

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
n_proc = comm.Get_size()


#======= repair functions =====
def FixFast(minutes):
    return True

def FixNever(minutes):
    return False

def FixGaussian(minutes, acstream, mu, sigma):
    """
    Return True if the line has been repaired.
    Args:
        minutes (float) : how long has the line been down
        mu, sigma (float): repair time is N(mu, sigma)
    """
    # spell it out...
    Z = (minutes-mu)/sigma
    u = acstream.rand()
    retval = u < scipy.norm.cdf(Z)
    return retval

#======= end repair functions =====
    
def _md_dict(epath):
    p = str(egret.__path__)
    l = p.find("'")
    r = p.find("'", l+1)
    egretrootpath = p[l+1:r]
    if epath[0] != os.sep:
        test_case = os.path.join(egretrootpath, epath)
    else:
        test_case = os.path.join(egretrootpath, epath[1:])

    # look at egret/data/model_data.py for the format specification of md_dict
    return create_ModelData(test_case)
    

####################################
def pysp2_callback(
    scenario_name,
    etree=None,
    solver=None,
    epath=None,
    tee=False,
    acstream=None,
    convex_relaxation=False,
    verbose=False,
    ramp_coeff=1000000,
    load_mismatch_cost=1000,
    q_load_mismatch_cost=None,
):
    """
    mpisppy signature for scenario creation.
    Then find a starting solution for the scenario if solver option is not None.
    Note that stage numbers are one-based.

    Args:
        scenario_name (str):
            Put the scenario number on the end 
        etree ():
            Default is None.
        solver (str):
            Solver to use.
        epath (str):
            Path to the egret data
        tee (bool):
            If True, displays solver output. Default is False.
        acstream ():
            Default is None.
        convex_relaxation (bool):
            If True, build the convex relaxation. Default is False.
        verbose (bool, optional):
            If True, display verbose output. Default is False.
        ramp_coeff (int, optional):
            Default is 1e6.
        load_mismatch_cost (float, optional):
            Default is 1000.
        q_load_mismatch_cost (float, optional):
            Deafult is load_mismatch_cost.

    Returns:
        scenario (pyo.ConcreteModel): the scenario instance

    Attaches:
        _enodes (ACtree nodes): a list of the ACtree tree nodes
        _egret_md (egret tuple with dict as [1]) egret model data

    """
    # pull the number off the end of the scenario name
    scen_num = sputils.extract_num(scenario_name)

    if q_load_mismatch_cost is None:
        q_load_mismatch_cost = load_mismatch_cost

    def lines_up_and_down(stage_md_dict, enode):
        # local routine to configure the lines in stage_md_dict for the scenario
        LinesDown = []
        for f in enode.FailedLines:
            LinesDown.append(f[0])
        for this_branch in stage_md_dict.elements("branch"):
            if this_branch[0] in enode.LinesUp:
                this_branch[1]["in_service"] = True
            elif this_branch[0] in LinesDown:
                this_branch[1]["in_service"] = False
            else:
                print("enode.LinesUp=", enode.LinesUp)
                print("enode.FailedLines=", enode.FailedLines)
                raise RuntimeError("Branch (line) {} neither up nor down in scenario {}".\
                               format(this_branch[0], scenario_name))
    def _egret_model(md_dict):
        # the exact acopf model is hard-wired here:
        md_dict.data['system']['load_mismatch_cost'] = load_mismatch_cost
        md_dict.data['system']['q_load_mismatch_cost'] = q_load_mismatch_cost
        if not convex_relaxation:
            pyomod, mdict = eac.create_riv_acopf_model(md_dict, 
                                            include_feasibility_slack=True)

        else:
            pyomod, mdict = eac_relax.create_soc_relaxation(md_dict, 
                                                include_feasibility_slack=True,
                                                use_linear_relaxation=False)
        return pyomod, mdict

    # pull the number off the end of the scenario name
    scen_num = sputils.extract_num(scenario_name)

    numstages = etree.NumStages
    enodes = etree.Nodes_for_Scenario(scen_num)
    full_scenario_model = pyo.ConcreteModel()
    full_scenario_model.stage_models = dict()

    # look at egret/data/model_data.py for the format specification of md_dict
    first_stage_md_dict = _md_dict(epath)
    generator_set = first_stage_md_dict.attributes("generator")
    generator_names = generator_set["names"]

    # the following creates the first stage model
    full_scenario_model.stage_models[1], model_dict = _egret_model(
        first_stage_md_dict)

    full_scenario_model.stage_models[1].obj.deactivate()
    setattr(full_scenario_model,
            "stage_models_"+str(1),
            full_scenario_model.stage_models[1]) 
    
    for stage in range(2, numstages+1):
        #print ("stage={}".format(stage))

        stage_md_dict = copy.deepcopy(first_stage_md_dict)
        #print ("debug: processing node {}".format(enodes[stage-1].Name))
        lines_up_and_down(stage_md_dict, enodes[stage-1])
        
        full_scenario_model.stage_models[stage], model_dict = \
            _egret_model(stage_md_dict)
        full_scenario_model.stage_models[stage].obj.deactivate()
        setattr(full_scenario_model,
                "stage_models_"+str(stage),
                full_scenario_model.stage_models[stage]) 

    def aggregate_ramping_rule(m):
        """
        We are adding ramping to the obj instead of a constraint for now
        because we may not have ramp limit data.
        """
        retval = 0
        for stage in range(1, numstages):
            retval += sum((full_scenario_model.stage_models[stage+1].pg[this_gen]\
                    - full_scenario_model.stage_models[stage].pg[this_gen])**2\
                   for this_gen in generator_names)
        return retval
    full_scenario_model.ramping = pyo.Expression(rule=aggregate_ramping_rule)

    full_scenario_model.objective = pyo.Objective(expr=\
                           ramp_coeff * full_scenario_model.ramping+\
                            sum(full_scenario_model.stage_models[stage].obj.expr\
                                    for stage in range(1,numstages+1)))
    
    inst = full_scenario_model
    # end code from PySP1
    
    node_list = list()

    parent_name = None
    for sm1, enode in enumerate(etree.Nodes_for_Scenario(scen_num)):
        stage = sm1 + 1
        if stage < etree.NumStages:
            node_list.append(scenario_tree.ScenarioNode(
                name=enode.Name,
                cond_prob=enode.CondProb,
                stage=stage,
                cost_expression=inst.stage_models[stage].obj,
                nonant_list=[inst.stage_models[stage].pg,
                             inst.stage_models[stage].qg],
                scen_model=inst, parent_name=parent_name))
            parent_name = enode.Name

    if verbose:
        print(f"\nScenario creation for {scenario_name} on rank {rank}"
              f"    FailedLines (line,minutes)={enode.FailedLines}")
            
    inst._mpisppy_node_list = node_list
    inst._mpisppy_probability = 1 / etree.numscens
    # solve it so subsequent code will have a good start
    if solver is not None:
        print(f"scenario creation callback is solving {scenario_name} on rank {rank}")
        solver.solve(inst, tee=tee)   #symbolic_solver_labels=True, keepfiles=True)

    # attachments
    inst._enodes = enodes
    inst._egret_md = first_stage_md_dict

    return inst

#=============================================================================
def scenario_denouement(rank, scenario_name, scenario):

    print("Solution for scenario=%s - rank=%d" % (scenario_name, rank))

    print("LINE OUTAGE INFORMATION")
    for enode in scenario._enodes:
        enode.pprint()

    stages = list(scenario.stage_models.keys())
    gens = sorted((g for g in getattr(scenario, "stage_models_"+str(1)).pg.keys()))

    for gen in gens:

        print("GEN: %4s PG:" % gen, end="")

        for stage in stages:
            current_val = pyo.value(getattr(scenario, "stage_models_"+str(stage)).pg[gen])
            if stage == stages[0]:
                print("%6.2f -->> " % current_val, end=" ")
            else:
                print("%6.2f" % (current_val-previous_val), end=" ")
            previous_val = current_val
        print("")

        print("GEN: %4s QG:" % gen, end="")

        for stage in stages:
            current_val = pyo.value(getattr(scenario, "stage_models_"+str(stage)).qg[gen])
            if stage == stages[0]:
                print("%6.2f -->> " % current_val, end=" ")
            else:
                print("%6.2f" % (current_val-previous_val), end=" ")
            previous_val = current_val
        print("")


    print("")

#=========================
if __name__ == "__main__":
    # as of April 27, 2020 __main__ has been updated only for EF
    print("EF only")
    import mpisppy.MPI as mpi
    n_proc = mpi.COMM_WORLD.Get_size()  # for error check
    # start options
    solver_name = "cplex"
    print(f"Solving with {solver_name}")
    solver = pyo.SolverFactory(solver_name)
    if "gurobi" in solver_name:
        solver.options["BarHomogeneous"] = 1

    casename = "pglib-opf-master/pglib_opf_case14_ieee.m"
    # pglib_opf_case3_lmbd.m
    # pglib_opf_case118_ieee.m
    # pglib_opf_case300_ieee.m
    # pglib_opf_case2383wp_k.m
    # pglib_opf_case2000_tamu.m
    # do not use pglib_opf_case89_pegase
    # this path starts at egret
    egret_path_to_data = "/thirdparty/"+casename

    if len(sys.argv) != 4:
        print ("Usage: python ccop_multistage bf1 bf2 bf3")
        print ("e.g., for three-stage use python ccop_multistage 2 3 0")
        exit(1)
    if int(sys.argv[3]) == 0:
        number_of_stages = 3
        stage_duration_minutes = [5, 15, 30]
        branching_factors = [int(sys.argv[1]), int(sys.argv[2])]
        allbutlast = branching_factors[0]
    else:
        number_of_stages = 4
        stage_duration_minutes = [5, 15, 30, 30]
        branching_factors = [int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])]
        allbutlast = branching_factors[0] * branching_factors[1]

    nscen = allbutlast * branching_factors[number_of_stages-2]
    seed = 1134
    a_line_fails_prob = 0.2
    repair_fct = FixFast
    verbose = False
    if allbutlast % n_proc != 0 and n_proc % allbutlast != 0:
        raise RuntimeError("allbutlast={} must divide n_proc={}, or vice-versa".\
                           format(allbutlast, n_proc))
    # TBD: comment this out and try -np 4 with bfs of 2 3 0 (strange sense error)
    if nscen % n_proc != 0:
        raise RuntimeError("nscen={} must be a multiple of n_proc={}".\
                           format(nscen, n_proc))

    convex_relaxation = True
    print(f"Convex relaxation = {convex_relaxation}")
    scenario_creator_kwargs = {
        "convex_relaxation": convex_relaxation,
        "solver": None if convex_relaxation else solver,
        "tee": False,  # For initialization solves
        "epath": egret_path_to_data,
    }
    md_dict = _md_dict(egret_path_to_data)

    if verbose and rank==0:
        print("start data dump")
        print(list(md_dict.elements("generator")))
        for this_branch in md_dict.elements("branch"): 
            print("TYPE=",type(this_branch))
            print("B=",this_branch)
            print("IN SERVICE=",this_branch[1]["in_service"])
        print("GENERATOR SET=",md_dict.attributes("generator"))
        print("end data dump")

    lines = list()
    for this_branch in md_dict.elements("branch"):
        lines.append(this_branch[0])

    acstream = np.random.RandomState()
    scenario_creator_kwargs["etree"] = ET.ACTree(number_of_stages,
                                 branching_factors,
                                 seed,
                                 acstream,
                                 a_line_fails_prob,
                                 stage_duration_minutes,
                                 repair_fct,
                                 lines)
    scenario_creator_kwargs["acstream"] = acstream
    scenario_names=["Scenario_"+str(i)\
                    for i in range(1,len(scenario_creator_kwargs["etree"].rootnode.ScenarioList)+1)]

    
    # end options
    ef = sputils.create_EF(scenario_names,
                             pysp2_callback,
                             scenario_creator_kwargs=scenario_creator_kwargs)
    ###solver.options["BarHomogeneous"] = 1
    if "gurobi" in solver_name:
        solver.options["BarHomogeneous"] = 1

    results = solver.solve(ef, tee=True)
    pyo.assert_optimal_termination(results)

    print('EF objective value:', pyo.value(ef.EF_Obj))
    sputils.ef_nonants_csv(ef, "vardump.csv")
    for (sname, smodel) in sputils.ef_scenarios(ef):
        print (sname)
        for stage in smodel.stage_models:
            print ("   Stage {}".format(stage))
            for gen in smodel.stage_models[stage].pg:
                 print ("      gen={} pg={}, qg={}"\
                        .format(gen,
                                pyo.value(smodel.stage_models[stage].pg[gen]),
                                pyo.value(smodel.stage_models[stage].qg[gen])))
            print ("   obj={}".format(pyo.value(smodel.objective)))
    print("EF objective value for case {}={}".\
          format(pyo.value(casename), pyo.value(ef.EF_Obj)))
