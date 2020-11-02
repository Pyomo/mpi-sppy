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
from mpisppy.extensions.extension import MultiPHExtension
from mpisppy.extensions.fixer import Fixer
from mpisppy.extensions.mipgapper import Gapper
from mpisppy.extensions.xhatlooper import XhatLooper
from mpisppy.extensions.xhatclosest import XhatClosest
from mpisppy.examples.sizes.sizes import scenario_creator, \
                                       scenario_denouement, \
                                       _rho_setter, \
                                       id_fix_list_fct

ScenCount = 10  # 3 or 10

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: python sizes_demo.py solvername")
        quit()
    PHoptions = {}
    PHoptions["solvername"] = sys.argv[1]
    PHoptions["asynchronousPH"] = False
    PHoptions["PHIterLimit"] = 2
    PHoptions["defaultPHrho"] = 1
    PHoptions["convthresh"] = 0.001
    PHoptions["subsolvedirectives"] = None
    PHoptions["verbose"] = False
    PHoptions["display_timing"] = True
    PHoptions["display_progress"] = True
    # one way to set up sub-problem solver options
    PHoptions["iter0_solver_options"] = {"mipgap": 0.01}
    # another way
    PHoptions["iterk_solver_options"] = {"mipgap": 0.005}
    PHoptions["xhat_looper_options"] =  {"xhat_solver_options":\
                                         PHoptions["iterk_solver_options"],
                                         "scen_limit": 3,
                                         "dump_prefix": "delme",
                                         "csvname": "looper.csv"}
    PHoptions["xhat_closest_options"] =  {"xhat_solver_options":\
                                         PHoptions["iterk_solver_options"],
                                         "csvname": "closest.csv"}
    PHoptions["xhat_specific_options"] =  {"xhat_solver_options":
                                           PHoptions["iterk_solver_options"],
                                           "xhat_scenario_dict": \
                                           {"ROOT": "Scenario3"},
                                           "csvname": "specific.csv"}
    fixoptions = {}
    fixoptions["verbose"] = True
    fixoptions["boundtol"] = 0.01
    fixoptions["id_fix_list_fct"] = id_fix_list_fct

    PHoptions["fixeroptions"] = fixoptions

    PHoptions["gapperoptions"] = {"verbose": True,
                   "mipgapdict": {0: 0.01,
                                  1: 0.009,
                                  5: 0.005,
                                 10: 0.001}}


    all_scenario_names = list()
    for sn in range(ScenCount):
        all_scenario_names.append("Scenario"+str(sn+1))
    # end hardwire

    ######### EF ########
    solver = pyo.SolverFactory(PHoptions["solvername"])

    ef = mpisppy.utils.sputils.create_EF(all_scenario_names,
                                   scenario_creator,
                                   creator_options={"cb_data": ScenCount})
    if 'persistent' in PHoptions["solvername"]:
        solver.set_instance(ef, symbolic_solver_labels=True)
    solver.options["mipgap"] = 0.01
    results = solver.solve(ef, tee=PHoptions["verbose"])
    print('EF objective value:', pyo.value(ef.EF_Obj))
    #mpisppy.utils.sputils.ef_nonants_csv(ef, "vardump.csv")
    #### first PH ####

    #####multi_ext = {"ext_classes": [Fixer, Gapper, XhatLooper, XhatClosest]}
    multi_ext = {"ext_classes": [Fixer, Gapper]}
    ph = mpisppy.opt.ph.PH(PHoptions,
                                all_scenario_names,
                                scenario_creator,
                                scenario_denouement,
                                cb_data=ScenCount,
                                rho_setter=_rho_setter, 
                                PH_extensions=MultiPHExtension,
                                PH_extension_kwargs=multi_ext,
    )
    
    conv, obj, tbound = ph.ph_main()
    if ph.rank == 0:
         print ("Trival bound =",tbound)

    print("Quitting early.")

    ############ test W and xbar writers and special joint reader  ############
    from mpisppy.utils.wxbarwriter import WXBarWriter
    
    newph = mpisppy.opt.ph.PH(PHoptions,
                                   all_scenario_names,
                                   scenario_creator,
                                   scenario_denouement,
                                   cb_data=ScenCount)

    PHoptions["W_and_xbar_writer"] =  {"Wcsvdir": "Wdir",
                                       "xbarcsvdir": "xbardir"}

    conv, obj, tbound = newph.ph_main()
    #####
    from mpisppy.utils.wxbarreader import WXBarReader
    
    newph = mpisppy.opt.ph.PH(PHoptions,
                                   all_scenario_names,
                                   scenario_creator,
                                   scenario_denouement,
                                   cb_data=ScenCount)

    PHoptions["W_and_xbar_reader"] =  {"Wcsvdir": "Wdir",
                                       "xbarcsvdir": "xbardir"}

    conv, obj, tbound = newph.ph_main()

    quit()
    ############################# test xhatspecific ###############
    from mpisppy.xhatspecific import XhatSpecific
    print ("... testing xhat specific....")
    newph = mpisppy.opt.ph.PH(PHoptions,
                                   all_scenario_names,
                                   scenario_creator,
                                   scenario_denouement,
                                   cb_data=ScenCount)

    PHoptions["xhat_specific_options"] =  {"xhat_solver_options":
                                           PHoptions["iterk_solver_options"],
                                           "xhat_scenario_dict": \
                                           {"ROOT": "Scenario3"},
                                           "csvname": "specific.csv"}

    conv = newph.ph_main(rho_setter=_rho_setter, 
                         PH_extensions=XhatSpecific)

    ######### bundles #########
    PHoptions["bundles_per_rank"] = 2
    PHoptions["verbose"] = False

    ph = mpisppy.opt.ph.PH(PHoptions,
                                all_scenario_names,
                                scenario_creator,
                                scenario_denouement,
                                cb_data=ScenCount)
    
    conv = ph.ph_main(rho_setter=_rho_setter)

    ### avg, min, max extension #####
    ph = mpisppy.opt.ph.PH(PHoptions, all_scenario_names,
                                scenario_creator, scenario_denouement,
                                cb_data=ScenCount)
    ph.PHoptions["PHIterLimit"] = 3

    from mpisppy.extensions.avgminmaxer import MinMaxAvg
    PHoptions["avgminmax_name"] =  "FirstStageCost"
    conv, obj, bnd = ph.ph_main(PH_extensions=MinMaxAvg,
                                PH_converger=None,
                                rho_setter=None)

    

