# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# updated 23 April 2020
# Serial (not cylinders)

import os
import sys
import pyomo.environ as pyo
import mpisppy.phbase
import mpisppy.opt.ph
import mpisppy.scenario_tree as scenario_tree
from mpisppy.extensions.extension import MultiExtension
from mpisppy.extensions.fixer import Fixer
from mpisppy.extensions.mipgapper import Gapper
from mpisppy.extensions.xhatlooper import XhatLooper
from mpisppy.extensions.xhatclosest import XhatClosest
from mpisppy.extensions.wtracker_extension import Wtracker_extension
from sizes import scenario_creator, \
                  scenario_denouement, \
                  _rho_setter, \
                  id_fix_list_fct

ScenCount = 10  # 3 or 10

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: python sizes_demo.py solver_name")
        quit()
    options = {}
    options["solver_name"] = sys.argv[1]
    options["asynchronousPH"] = False
    options["PHIterLimit"] = 5
    options["defaultPHrho"] = 1
    options["convthresh"] = 0.001
    options["subsolvedirectives"] = None
    options["verbose"] = False
    options["display_timing"] = True
    options["display_progress"] = True
    options["linearize_proximal_terms"] = True
    # one way to set up sub-problem solver options
    options["iter0_solver_options"] = {"mipgap": 0.01}
    # another way
    options["iterk_solver_options"] = {"mipgap": 0.005}
    options["xhat_looper_options"] =  {"xhat_solver_options":\
                                         options["iterk_solver_options"],
                                         "scen_limit": 3,
                                         "dump_prefix": "delme",
                                         "csvname": "looper.csv"}
    options["xhat_closest_options"] =  {"xhat_solver_options":\
                                         options["iterk_solver_options"],
                                         "csvname": "closest.csv"}
    options["xhat_specific_options"] =  {"xhat_solver_options":
                                           options["iterk_solver_options"],
                                           "xhat_scenario_dict": \
                                           {"ROOT": "Scenario3"},
                                           "csvname": "specific.csv"}
    fixoptions = {}
    fixoptions["verbose"] = True
    fixoptions["boundtol"] = 0.01
    fixoptions["id_fix_list_fct"] = id_fix_list_fct

    options["fixeroptions"] = fixoptions

    options["gapperoptions"] = {"verbose": True,
                   "mipgapdict": {0: 0.01,
                                  1: 0.009,
                                  5: 0.005,
                                 10: 0.001}}
    options["wtracker_options"] ={"wlen": 4,
                                  "reportlen": 6,
                                  "stdevthresh": 0.1}
    

    all_scenario_names = list()
    for sn in range(ScenCount):
        all_scenario_names.append("Scenario"+str(sn+1))
    # end hardwire

    ######### EF ########
    solver = pyo.SolverFactory(options["solver_name"])

    ef = mpisppy.utils.sputils.create_EF(
        all_scenario_names,
        scenario_creator,
        scenario_creator_kwargs={"scenario_count": ScenCount},
    )
    if 'persistent' in options["solver_name"]:
        solver.set_instance(ef, symbolic_solver_labels=True)
    solver.options["mipgap"] = 0.01
    results = solver.solve(ef, tee=options["verbose"])
    print('EF objective value:', pyo.value(ef.EF_Obj))
    #mpisppy.utils.sputils.ef_nonants_csv(ef, "vardump.csv")
    #### first PH ####

    #####multi_ext = {"ext_classes": [Fixer, Gapper, XhatLooper, XhatClosest]}
    multi_ext = {"ext_classes": [Fixer, Gapper, Wtracker_extension]}
    ph = mpisppy.opt.ph.PH(
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement,
        scenario_creator_kwargs={"scenario_count": ScenCount},
        rho_setter=_rho_setter, 
        extensions=MultiExtension,
        extension_kwargs=multi_ext,
    )
    
    conv, obj, tbound = ph.ph_main()
    if ph.cylinder_rank == 0:
         print ("Trival bound =",tbound)

    #print("Quitting early.")
    #quit()

    ############ test W and xbar writers and special joint reader  ############
    from mpisppy.utils.wxbarwriter import WXBarWriter
    
    newph = mpisppy.opt.ph.PH(
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement,
        scenario_creator_kwargs={"scenario_count": ScenCount},
    )

    options["W_and_xbar_writer"] =  {"Wcsvdir": "Wdir",
                                       "xbarcsvdir": "xbardir"}

    conv, obj, tbound = newph.ph_main()
    #####
    from mpisppy.utils.wxbarreader import WXBarReader
    
    newph = mpisppy.opt.ph.PH(
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement,
        scenario_creator_kwargs={"scenario_count": ScenCount},
    )

    options["W_and_xbar_reader"] =  {"Wcsvdir": "Wdir",
                                       "xbarcsvdir": "xbardir"}

    conv, obj, tbound = newph.ph_main()

    quit()
    ############################# test xhatspecific ###############
    from mpisppy.xhatspecific import XhatSpecific
    print ("... testing xhat specific....")
    newph = mpisppy.opt.ph.PH(
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement,
        scenario_creator_kwargs={"scenario_count": ScenCount},
    )

    options["xhat_specific_options"] =  {"xhat_solver_options":
                                           options["iterk_solver_options"],
                                           "xhat_scenario_dict": \
                                           {"ROOT": "Scenario3"},
                                           "csvname": "specific.csv"}

    conv = newph.ph_main(rho_setter=_rho_setter, 
                         extensions=XhatSpecific)

    ######### bundles #########
    options["bundles_per_rank"] = 2
    options["verbose"] = False

    ph = mpisppy.opt.ph.PH(
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement,
        scenario_creator_kwargs={"scenario_count": ScenCount},
    )
    
    conv = ph.ph_main(rho_setter=_rho_setter)

    ### avg, min, max extension #####
    ph = mpisppy.opt.ph.PH(
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement,
        scenario_creator_kwargs={"scenario_count": ScenCount},
        extensions=MinMaxAvg,
        ph_converger=None,
        rho_setter=None,
    )
    ph.options["PHIterLimit"] = 3

    from mpisppy.extensions.avgminmaxer import MinMaxAvg
    options["avgminmax_name"] =  "FirstStageCost"
    conv, obj, bnd = ph.ph_main()

    

