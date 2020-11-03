# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# four stage test
# mpiexec -np 8 python -m mpi4py forustage.py 2 2 2 1 0
# (see the first lines of main() to change instances)
import os
import numpy as np
# Hub and spoke SPBase classes
from mpisppy.phbase import PHBase
from mpisppy.opt.ph import PH
# Hub and spoke SPCommunicator classes
from mpisppy.cylinders.xhatspecific_bounder import XhatSpecificInnerBound
from mpisppy.cylinders.hub import PHHub
# Make it all go
from mpisppy.utils.sputils import spin_the_wheel
# the problem
import mpisppy.examples.acopf3.ACtree as etree
from mpisppy.examples.acopf3.ccopf_multistage import pysp2_callback,\
    scenario_denouement, _md_dict, FixFast, FixNever, FixGaussian
import  mpisppy.examples.acopf3.rho_setter as rho_setter

import pyomo.environ as pyo
import socket
import sys
import datetime as dt

import mpi4py.MPI as mpi
comm_global = mpi.COMM_WORLD
rank_global = comm_global.Get_rank()
n_proc = comm_global.Get_size()

# =========================


def main():
    start_time = dt.datetime.now()
    # start options
    casename = "pglib-opf-master/pglib_opf_case14_ieee.m"
    # pglib_opf_case3_lmbd.m
    # pglib_opf_case118_ieee.m
    # pglib_opf_case300_ieee.m
    # pglib_opf_case2383wp_k.m
    # pglib_opf_case2000_tamu.m
    # do not use pglib_opf_case89_pegase
    egret_path_to_data = "/thirdparty/"+casename
    number_of_stages = 4
    stage_duration_minutes = [5, 15, 30, 30]
    if len(sys.argv) != number_of_stages + 3:
        print ("Usage: python fourstage.py bf1 bf2 bf3 iters scenperbun(!)")
        exit(1)
    branching_factors = [int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])]
    PHIterLimit = int(sys.argv[4])
    scenperbun = int(sys.argv[5])
    solvername = sys.argv[6]

    seed = 1134
    a_line_fails_prob = 0.1   # try 0.2 for more excitement
    repair_fct = FixFast
    verbose = False
    # end options
    # create an arbitrary xhat for xhatspecific to use
    xhat_dict = {"ROOT": "Scenario_1"}
    for i in range(branching_factors[0]):
        st1scen = 1 + i*(branching_factors[1] + branching_factors[2])
        xhat_dict["ROOT_"+str(i)] = "Scenario_"+str(st1scen)
        for j in range(branching_factors[2]):
            st2scen = st1scen + j*branching_factors[2]
            xhat_dict["ROOT_"+str(i)+'_'+str(j)] = "Scenario_"+str(st2scen)

    cb_data = dict()
    cb_data["convex_relaxation"] = True
    cb_data["epath"] = egret_path_to_data
    if cb_data["convex_relaxation"]:
        # for initialization solve
        solvername = solvername
        solver = pyo.SolverFactory(solvername)
        cb_data["solver"] = None
        ##if "gurobi" in solvername:
            ##solver.options["BarHomogeneous"] = 1
    else:
        solvername = "ipopt"
        solver = pyo.SolverFactory(solvername)
        if "gurobi" in solvername:
            solver.options["BarHomogeneous"] = 1
        cb_data["solver"] = solver
    md_dict = _md_dict(cb_data)

    if verbose:
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
        
    cb_data["etree"] = etree.ACTree(number_of_stages,
                                    branching_factors,
                                    seed,
                                    acstream,
                                    a_line_fails_prob,
                                    stage_duration_minutes,
                                    repair_fct,
                                    lines)
    cb_data["acstream"] = acstream

    all_scenario_names=["Scenario_"+str(i)\
                        for i in range(1,len(cb_data["etree"].\
                                             rootnode.ScenarioList)+1)]
    all_nodenames = cb_data["etree"].All_Nonleaf_Nodenames()

    PHoptions = dict()
    if cb_data["convex_relaxation"]:
        PHoptions["solvername"] = solvername
        if "gurobi" in PHoptions["solvername"]:
            PHoptions["iter0_solver_options"] = {"BarHomogeneous": 1}
            PHoptions["iterk_solver_options"] = {"BarHomogeneous": 1}
        else:
            PHoptions["iter0_solver_options"] = None
            PHoptions["iterk_solver_options"] = None
    else:
        PHoptions["solvername"] = "ipopt"
        PHoptions["iter0_solver_options"] = None
        PHoptions["iterk_solver_options"] = None
    PHoptions["PHIterLimit"] = PHIterLimit
    PHoptions["defaultPHrho"] = 1
    PHoptions["convthresh"] = 0.001
    PHoptions["subsolvedirectives"] = None
    PHoptions["verbose"] = False
    PHoptions["display_timing"] = False
    PHoptions["display_progress"] = True
    PHoptions["iter0_solver_options"] = None
    PHoptions["iterk_solver_options"] = None
    PHoptions["branching_factors"] = branching_factors

    # try to do something interesting for bundles per rank
    if scenperbun > 0:
        nscen = branching_factors[0] * branching_factors[1]
        PHoptions["bundles_per_rank"] = int((nscen / n_proc) / scenperbun)
    if rank_global == 0:
        appfile = "acopf.app"
        if not os.path.isfile(appfile):
            with open(appfile, "w") as f:
                f.write("datetime, hostname, BF1, BF2, seed, solver, n_proc")
                f.write(", bunperank, PHIterLimit, convthresh")
                f.write(", PH_IB, PH_OB")
                f.write(", PH_lastiter, PH_wallclock, aph_frac_needed")
                f.write(", APH_IB, APH_OB")
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


    # PH hub
    PHoptions["tee-rank0-solves"] = True
    hub_dict = {
        "hub_class": PHHub,
        "hub_kwargs": {"options": None},
        "opt_class": PH,
        "opt_kwargs": {
            "PHoptions": PHoptions,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": pysp2_callback,
            'scenario_denouement': scenario_denouement,
            "cb_data": cb_data,
            "rho_setter": rho_setter.ph_rhosetter_callback,
            "PH_extensions": None,
            "all_nodenames":all_nodenames,
        }
    }

    xhat_options = PHoptions.copy()
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhat_options["xhat_specific_options"] = {"xhat_solver_options":
                                          PHoptions["iterk_solver_options"],
                                          "xhat_scenario_dict": xhat_dict,
                                          "csvname": "specific.csv"}

    ub2 = {
        'spoke_class': XhatSpecificInnerBound,
        "spoke_kwargs": dict(),
        "opt_class": PHBase,   
        'opt_kwargs': {
            'PHoptions': xhat_options,
            'all_scenario_names': all_scenario_names,
            'scenario_creator': pysp2_callback,
            'scenario_denouement': scenario_denouement,
            "cb_data": cb_data,
            'all_nodenames': all_nodenames
        },
    }
    
    list_of_spoke_dict = [ub2]
    spcomm, opt_dict = spin_the_wheel(hub_dict, list_of_spoke_dict)
    if "hub_class" in opt_dict:  # we are hub rank
        if spcomm.opt.rank == spcomm.opt.rank0:  # we are the reporting hub rank
            ph_end_time = dt.datetime.now()
            IB = spcomm.BestInnerBound
            OB = spcomm.BestOuterBound
            print("BestInnerBound={} and BestOuterBound={}".\
                  format(IB, OB))
            with open(appfile, "a") as f:
                f.write(", "+str(IB)+", "+str(OB)+", "+str(spcomm.opt._PHIter))
                f.write(", "+str((ph_end_time - start_time).total_seconds()))

                
if __name__=='__main__':
    main()
