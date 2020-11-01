# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# update April 2020: BUT this really needs upper and lower bound spokes
# dlw February 2019: PySP 2 for the sslp example

import os
import sys
import socket
import datetime as dt
import pyomo.environ as pyo
import mpisppy.opt.ph
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import mpisppy.examples.sslp.model.ReferenceModel as ref
import mpisppy.extensions.fixer as fixer


def scenario_creator(scenario_name, node_names=None, cb_data=None):
    """ The callback needs to create an instance and then attach
        the PySP nodes to it in a list _PySPnode_list ordered by stages.
        Optionally attach _PHrho.
    """
    datadir = cb_data
    fname = datadir + os.sep + scenario_name + ".dat"
    model = ref.model.create_instance(fname, name=scenario_name)

    # now attach the one and only tree node (ROOT is a reserved word)
    model._PySPnode_list = [
        scenario_tree.ScenarioNode(
            "ROOT", 1.0, 1, model.FirstStageCost, None, [model.FacilityOpen], model
        )
    ]
    return model


def scenario_denouement(rank, scenario_name, scenario):
    pass


def id_fix_list_fct(s):
    """ specify tuples used by the fixer.

        Args:
            s (ConcreteModel): the sizes instance.
        Returns:
             i0, ik (tuples): one for iter 0 and other for general iterations.
                 Var id,  threshold, nb, lb, ub
                 The threshold is on the square root of the xbar squared differnce
                 nb, lb an bu an "no bound", "upper" and "lower" and give the numver
                     of iterations or None for ik and for i0 anything other than None
                     or None. In both cases, None indicates don't fix.
        Note:
            This is just here to provide an illustration, we don't run long enough.
    """

    # iter0tuples = [
    #     fixer.Fixer_tuple(s.FacilityOpen[i], th=None, nb=None, lb=None, ub=None)
    #     for i in s.FacilityOpen
    # ]
    iterktuples = [
        fixer.Fixer_tuple(s.FacilityOpen[i], th=0, nb=None, lb=20, ub=20)
        for i in s.FacilityOpen
    ]
    return None, iterktuples


if __name__ == "__main__":
    msg = (
        "Give instance name, then bundles per rank, then PH iters "
        + "then rho (e.g., sslp_15_45_5 0 6 1)"
    )

    if len(sys.argv) != 5:
        print(msg)
        quit()
    instname = sys.argv[1]
    datadir = "data" + os.sep + instname + os.sep + "scenariodata"
    if not os.path.isdir(datadir):
        print(msg, "\n   bad instance name=", instname)
        quit()
    try:
        bunper = int(sys.argv[2])
    except:
        print(msg, "\n    bad number of bundles per rank=", sys.argv[2])
        quit()
    try:
        maxit = int(sys.argv[3])
    except:
        print(msg, "\n    bad max iterations=", sys.argv[3])
        quit()
    try:
        rho = int(sys.argv[4])
    except:
        print(msg, "\n    bad rho=", sys.argv[4])
        quit()

    # The number of scenarios is the last number in the instance name
    ScenCount = sputils.extract_num(instname)

    start_time = dt.datetime.now()

    PHoptions = {}
    PHoptions["solvername"] = "gurobi_persistent"
    PHoptions["PHIterLimit"] = maxit
    PHoptions["defaultPHrho"] = rho
    PHoptions["convthresh"] = -1
    PHoptions["subsolvedirectives"] = None
    PHoptions["verbose"] = False
    PHoptions["display_timing"] = False
    PHoptions["display_progress"] = True
    ### async section ###
    PHoptions["asynchronous"] = True
    PHoptions["async_frac_needed"] = 0.5
    PHoptions["async_sleep_secs"] = 1
    ### end asyn section ###
    # one way to set up sub-problem solver options
    PHoptions["iter0_solver_options"] = {"mipgap": 0.01}
    # another way
    PHoptions["iterk_solver_options"] = {"mipgap": 0.02, "threads": 4}
    PHoptions["xhat_solver_options"] = PHoptions["iterk_solver_options"]
    if bunper > 0:
        PHoptions["bundles_per_rank"] = bunper
    PHoptions["append_file_name"] = "sslp.app"

    fixoptions = {}
    fixoptions["verbose"] = True
    fixoptions["boundtol"] = 0.01
    fixoptions["id_fix_list_fct"] = id_fix_list_fct

    PHoptions["fixeroptions"] = fixoptions

    PHoptions["gapperoptions"] = {
        "verbose": True,
        "mipgapdict": {0: 0.02, 1: 0.02, 5: 0.01, 10: 0.005},
    }

    all_scenario_names = list()
    for sn in range(ScenCount):
        all_scenario_names.append("Scenario" + str(sn + 1))

    ph = mpisppy.opt.ph.PH(
        PHoptions,
        all_scenario_names,
        scenario_creator,
        scenario_denouement,
        cb_data=datadir,
    )

    if ph.rank == 0:
        appfile = PHoptions["append_file_name"]
        if not os.path.isfile(appfile):
            with open(appfile, "w") as f:
                f.write("datetime, hostname, instname, solver, n_proc")
                f.write(", bunperank, PHIterLimit, convthresh, Rho")
                f.write(", xhatobj, bound, trivialbnd, lastiter, wallclock")
                f.write(", asyncfrac, asyncsleep")
        if "bundles_per_rank" in PHoptions:
            nbunstr = str(PHoptions["bundles_per_rank"])
        else:
            nbunstr = "0"
        with open(appfile, "a") as f:
            f.write(
                "\n" + str(start_time) + "," + socket.gethostname() + "," + instname
            )
            f.write(", " + str(PHoptions["solvername"]))
            f.write(", " + str(ph.n_proc))
            f.write(", " + nbunstr)
            f.write(", " + str(PHoptions["PHIterLimit"]))
            f.write(", " + str(PHoptions["convthresh"]))
            f.write(", " + str(PHoptions["defaultPHrho"]))
    ###from mpisppy.xhatlooper import XhatLooper
    conv, eobj, tbound = ph.ph_main()

    # PH_extensions=XhatLooper,

    print("\nQUITTING EARLY; this needs to be a hub and have spokes!!!")
    quit()

    dopts = {"mipgap": 0.001}
    ph.PHoptions["asynchronous"] = False
    objbound = ph.post_solve_bound(solver_options=dopts, verbose=False)
    if ph.rank == 0:
        print("**** Lagrangian objective function bound=", objbound)

    end_time = dt.datetime.now()

    if ph.rank == 0:
        with open(appfile, "a") as f:
            f.write(", " + str(objbound) + ", " + str(tbound) + ", " + str(ph._PHIter))
            f.write(", " + str((end_time - start_time).total_seconds()))
            if PHoptions["asynchronous"]:
                f.write(", " + str(PHoptions["async_frac_needed"]))
                f.write(", " + str(PHoptions["async_sleep_secs"]))
