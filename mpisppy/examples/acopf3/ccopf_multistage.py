# This software is distributed under the 3-clause BSD License.
# JPW and DLW; July 2019; ccopf create scenario instances for line outages
# extended Fall 2019 by DLW
import egret
import egret.models.acopf as eac
from egret.data.model_data import ModelData
from egret.parsers.matpower_parser import create_ModelData
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import mpisppy.opt.ph
import mpisppy.examples.acopf3.ACtree as etree
import mpisppy.opt.aph
import mpisppy.examples.acopf3.rho_setter as rho_setter

import os
import sys
import copy
import scipy
import socket
import numpy as np
import datetime as dt
import mpi4py.MPI as mpi

import pyomo.environ as pyo

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
n_proc = comm.Get_size()


#======= repair functions =====
def FixFast(minutes):
    return True

def FixNever(minutes):
    return False

def FixGaussian(minutes, mu, sigma):
    """
    Return True if the line has been repaired.
    Args:
        minutes (float) : how long has the line been down
        mu, sigma (float): repair time is N(mu, sigma)
    """
    # spell it out...
    Z = (minutes-mu)/sigma
    u = np.random.rand()
    retval = u < scipy.norm.cdf(Z)
    return retval

#======= end repair functions =====
    
def _md_dict(cb_data):
    p = str(egret.__path__)
    l = p.find("'")
    r = p.find("'", l+1)
    egretrootpath = p[l+1:r]
    test_case = egretrootpath+cb_data["epath"]

    # look at egret/data/model_data.py for the format specification of md_dict
    return create_ModelData(test_case)
    

####################################
def pysp2_callback(scenario_name,
                    node_names=None,
                    cb_data=None):
    """
    mpisppy signature for scenario creation. Basically, just call the PySP1 fct.
    Then find a starting solution for the scenario if solver option is not None.
    Note that stage numbers are one-based.

    Args:
        scenario_name (str): put the scenario number on the end 
        node_names (int): not used
        cb_data: (dict) "etree", "solver", "epath", "tee"

    Returns:
        scenario (pyo.ConcreteModel): the scenario instance

    Attaches:
        _enodes (ACtree nodes): a list of the ACtree tree nodes
        _egret_md (egret tuple with dict as [1]) egret model data

    """
    # pull the number off the end of the scenario name
    scen_num = sputils.extract_num(scenario_name)

    etree = cb_data["etree"]
    solver = cb_data["solver"]
    """
    inst = pysp_instance_creation_callback(scenario_tree_model = etree,
                                           scenario_name = scenario_name)
    """
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
    
    # pull the number off the end of the scenario name
    scen_num = sputils.extract_num(scenario_name)
    #print ("debug scen_num=",scen_num)

    numstages = etree.NumStages
    enodes = etree.Nodes_for_Scenario(scen_num)
    full_scenario_model = pyo.ConcreteModel()
    full_scenario_model.stage_models = dict()

    # the exact acopf model is hard-wired here:
    acopf_model = eac.create_riv_acopf_model

    # look at egret/data/model_data.py for the format specification of md_dict
    first_stage_md_dict = _md_dict(cb_data)
    generator_set = first_stage_md_dict.attributes("generator")
    generator_names = generator_set["names"]

    # the following creates the first stage model
    full_scenario_model.stage_models[1], model_dict = acopf_model(
        first_stage_md_dict, include_feasibility_slack=True)
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
            acopf_model(stage_md_dict, include_feasibility_slack=True)
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
                            1000000.0 * full_scenario_model.ramping+\
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
                scen_name_list=enode.ScenarioList,
                nonant_list=[inst.stage_models[stage].pg,
                             inst.stage_models[stage].qg],
                scen_model=inst, parent_name=parent_name))
            parent_name = enode.Name
    
    inst._PySPnode_list = node_list
    # Optionally assign probability to PySP_prob 
    inst.PySP_prob = 1 / etree.numscens
    # solve it so subsequent code will have a good start
    if solver is not None:
        solver.solve(inst)

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
    print("EF only but you still have to give iters and bundles")
    import mpi4py.MPI as mpi
    n_proc = mpi.COMM_WORLD.Get_size()  # for error check
    # start options
    solvername = "ipopt"
    solver = pyo.SolverFactory(solvername)
    casename = "pglib-opf-master/pglib_opf_case118_ieee.m"
    # pglib_opf_case3_lmbd.m
    # pglib_opf_case118_ieee.m
    # pglib_opf_case300_ieee.m
    # pglib_opf_case2383wp_k.m
    # pglib_opf_case2000_tamu.m
    # do not use pglib_opf_case89_pegase
    # this path starts at egret
    egret_path_to_data = "/thirdparty/"+casename
    number_of_stages = 3
    stage_duration_minutes = [5, 15, 30]

    if len(sys.argv) != number_of_stages + 2:
        print ("Usage: python ccop_multistage bf1 bf2 iters scenperbun(!)")
        exit(1)
    branching_factors = [int(sys.argv[1]), int(sys.argv[2])]
    PHIterLimit = int(sys.argv[3])
    scenperbun = int(sys.argv[4])

    seed = 1134
    a_line_fails_prob = 0.2
    repair_fct = FixFast
    verbose = False
    if branching_factors[0] % n_proc != 0 and n_proc % branching_factors[0] != 0:
        raise RuntimeError("bf1={} must divide n_proc={}, or vice-versa".\
                           format(branching_factors[0], n_proc))
    nscen = branching_factors[0] * branching_factors[1]
    # TBD: comment this out and try -np 4 with bfs of 2 3 (strange sense error)
    if nscen % n_proc != 0:
        raise RuntimeError("nscen={} must be a multiple of n_proc={}".\
                           format(nscen, n_proc))
    cb_data = dict()
    cb_data["solver"] = solver # can be None
    cb_data["tee"] = False # for inialization solve
    cb_data["epath"] = egret_path_to_data
    md_dict = _md_dict(cb_data)

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
    
    cb_data["etree"] = etree.ACTree(number_of_stages,
                                  branching_factors,
                                  seed,
                                  a_line_fails_prob,
                                  stage_duration_minutes,
                                  repair_fct,
                                  lines)
    cb_data["epath"] = egret_path_to_data

    creator_options = {"cb_data": cb_data}
    scenario_names=["Scenario_"+str(i)\
                    for i in range(1,len(cb_data["etree"].rootnode.ScenarioList)+1)]

    
    # end options
    ef = sputils.create_EF(scenario_names,
                             pysp2_callback,
                             creator_options)
    results = solver.solve(ef, tee=True)
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

    quit()


    # Junk from here down.... as of April 27, 2020
    # create an arbitrary xhat for xhatspecific to use
    xhat_dict = {"ROOT": "Scenario_1"}
    for i in range(branching_factors[0]):
        xhat_dict["ROOT_"+str(i)] = "Scenario_"+str(1 + i*branching_factors[1])
    
    doef = False
    if doef:
        ''' solve the EF '''
    start_time = dt.datetime.now()
    
    ''' "solve" using PH '''
    print ("starting ph/aph work")
    all_nodenames = cb_data["etree"].All_Nonleaf_Nodenames()
    
    PHoptions = dict()
    PHoptions["solvername"] = "ipopt"
    PHoptions["PHIterLimit"] = PHIterLimit
    PHoptions["defaultPHrho"] = 1
    PHoptions["convthresh"] = 0.001
    PHoptions["subsolvedirectives"] = None
    PHoptions["verbose"] = False
    PHoptions["display_timing"] = True
    PHoptions["display_progress"] = True
    PHoptions["iter0_solver_options"] = None
    PHoptions["iterk_solver_options"] = None
    PHoptions["branching_factors"] = branching_factors
    """
    PHoptions["xhat_looper_options"] =  {"xhat_solver_options":\
                                         PHoptions["iterk_solver_options"],
                                         "scen_limit": 3,
                                         "dump_prefix": "delme",
                                         "csvname": "looper.csv"}
    """
    xhat_solver_options = dict()
    xhat_solver_options ["Tee"] = True # special for Xhat
    PHoptions["xhat_specific_options"] = {"xhat_solver_options":
                                          xhat_solver_options,
                                          "xhat_scenario_dict": xhat_dict,
                                          "csvname": "specific.csv"}
    # try to do something interesting for bundles per rank
    if scenperbun > 0:
        PHoptions["bundles_per_rank"] = int((nscen / n_proc) / scenperbun)
    if rank == 0:
        appfile = "acopf.app"
        if not os.path.isfile(appfile):
            with open(appfile, "w") as f:
                f.write("datetime, hostname, BF1, BF2, seed, solver, n_proc")
                f.write(", bunperank, PHIterLimit, convthresh")
                f.write(", PH_xhatobj, trivialbnd")
                f.write(", PH_lastiter, PH_wallclock, aph_frac_needed")
                f.write(", APH_xhatobj, trivialbnd")
                f.write(", APH_lastiter, APH_wallclock")
        if "bundles_per_rank" in PHoptions:
            nbunstr = str(PHoptions["bundles_per_rank"])
        else:
            nbunstr = "0"
        oline = "\n"+ str(start_time)+","+socket.gethostname()
        oline += ","+str(branching_factors[0])+","+str(branching_factors[1])
        oline += ", "+str(seed) + ", "+str(PHoptions["solvername"])
        oline += ", "+str(n_proc) + ", "+ nbunstr
        oline += ", "+str(PHoptions["PHIterLimit"])
        oline += ", "+str(PHoptions["convthresh"])

        with open(appfile, "a") as f:
            f.write(oline)
        print(oline)

    doph = True
    if doph:
        ph = mpisppy.opt.ph.PH(PHoptions,
                                  scenario_names,
                                  pysp2_callback,
                                  scenario_denouement,
                                  all_nodenames=all_nodenames)

        conv, obj, tbound = ph.ph_main(rho_setter=rho_setter.ph_rhosetter_callback, 
                                       PH_extensions = XhatSpecific,
                                       cb_data = cb_data)
        phxhat = str(ph._xhat_specific_obj_final)
        phLastIter = str(ph._PHIter)
        if rank == 0:
             print ("Trival bound =",tbound)
    else:
        phxhat = "N/A"
        phLastIter = "N/A"
        tbound = "N/A"

    ###ph._disable_W_and_prox()
    ###e_obj = ph.Eobjective()
    if rank == 0:
         ###print ("unweighted e_obj={}".format(e_obj))
         print ("x_hat obj={}".format(phxhat))
    ph_end_time = dt.datetime.now()
    if rank == 0:
        with open(appfile, "a") as f:
            f.write(", "+phxhat
                    +", "+str(tbound)+", "+phLastIter)
            f.write(", "+str((ph_end_time - start_time).total_seconds()))

    """ aph """
    print ("staring aph work")
    PHoptions["async_frac_needed"] = 0.5
    PHoptions["async_sleep_secs"] = 0.5
    if rank == 0:
        with open(appfile, "a") as f:
            f.write(", "+str(PHoptions["async_frac_needed"]))
    aph_start_time = dt.datetime.now()
    aph = mpisppy.opt.aph.APH(PHoptions,
                        scenario_names,
                        pysp2_callback,
                        scenario_denouement,
                        all_nodenames=all_nodenames)
    
    conv, obj, tbound = aph.APH_main(rho_setter=None, 
                                     PH_extensions = XhatSpecific,
                                     cb_data = cb_data)
    aph_end_time = dt.datetime.now()
    if rank == 0:
        with open(appfile, "a") as f:
            f.write(", "+str(aph._xhat_specific_obj_final)\
                    +", "+str(tbound)+", "+str(aph._PHIter))
            f.write(", "+str((aph_end_time - aph_start_time).total_seconds()))
