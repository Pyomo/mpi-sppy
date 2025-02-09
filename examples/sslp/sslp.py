###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# update April 2020: BUT this really needs upper and lower bound spokes
# dlw February 2019: PySP 2 for the sslp example

import os
import sys
import socket
import datetime as dt
import mpisppy.opt.ph
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import mpisppy.extensions.fixer as fixer

import model.ReferenceModel as ref


from mpisppy.convergers.primal_dual_converger import PrimalDualConverger


def scenario_creator(scenario_name, data_dir=None, surrogate=False):
    """ The callback needs to create an instance and then attach
        the PySP nodes to it in a list _mpisppy_node_list ordered by stages.
        Optionally attach _PHrho.
    """
    if data_dir is None:
        raise ValueError("kwarg `data_dir` is required for SSLP scenario_creator")
    fname = data_dir + os.sep + scenario_name + ".dat"
    model = ref.model.create_instance(fname, name=scenario_name)

    if surrogate:
        model.total_facilities = ref.Var(within=ref.NonNegativeIntegers, bounds=(0, model.NumServers))

        @model.Constraint()
        def total_facilities_constr(m):
            return (0, sum(m.FacilityOpen.values()) - m.total_facilities)

        surrogate_nonant_list = [model.total_facilities,]
    else:
        surrogate_nonant_list = []

    # now attach the one and only tree node (ROOT is a reserved word)
    model._mpisppy_node_list = [
        scenario_tree.ScenarioNode(
            "ROOT", 1.0, 1, model.FirstStageCost, [model.FacilityOpen], model, surrogate_nonant_list=surrogate_nonant_list
        )
    ]
    model._mpisppy_probability = "uniform"

    return model

def scenario_denouement(rank, scenario_name, scenario):
    pass


########## helper functions ########

#=========
def scenario_names_creator(num_scens,start=None):
    # one-based scenarios
    # if start!=None, the list starts with the 'start' labeled scenario
    if (start is None) :
        start=1
    return [f"Scenario{i}" for i in range(start, start+num_scens)]


#=========
def inparser_adder(cfg):
    # add options unique to sizes
    # we don't want num_scens from the command line
    cfg.mip_options()
    cfg.add_to_config("instance_name",
                        description="sslp instance name (e.g., sslp_15_45_10)",
                        domain=str,
                        default=None)                
    cfg.add_to_config("sslp_data_path",
                        description="path to sslp data (e.g., ./data)",
                        domain=str,
                        default=None)                


#=========
def kw_creator(cfg):
    # linked to the scenario_creator and inparser_adder
    # side-effect is dealing with num_scens
    inst = cfg.instance_name
    ns = int(inst.split("_")[-1])
    if hasattr(cfg, "num_scens"):
        if cfg.num_scens != ns:
            raise RuntimeError(f"Argument num-scens={cfg.num_scens} does not match the number "
                               "implied by instance name={ns} "
                               "\n(--num-scens is not needed for sslp)")
    else:
        cfg.add_and_assign("num_scens","number of scenarios", int, None, ns)
    data_dir = os.path.join(cfg.sslp_data_path, inst, "scenariodata")
    kwargs = {"data_dir": data_dir}
    return kwargs


def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    """ Create a scenario within a sample tree. Mainly for multi-stage and simple for two-stage.
        (this function supports zhat and confidence interval code)
    Args:
        sname (string): scenario name to be created
        stage (int >=1 ): for stages > 1, fix data based on sname in earlier stages
        sample_branching_factors (list of ints): branching factors for the sample tree
        seed (int): To allow random sampling (for some problems, it might be scenario offset)
        given_scenario (Pyomo concrete model): if not None, use this to get data for ealier stages
        scenario_creator_kwargs (dict): keyword args for the standard scenario creator funcion
    Returns:
        scenario (Pyomo concrete model): A scenario for sname with data in stages < stage determined
                                         by the arguments
    """
    # Since this is a two-stage problem, we don't have to do much.
    sca = scenario_creator_kwargs.copy()
    sca["seedoffset"] = seed
    sca["num_scens"] = sample_branching_factors[0]  # two-stage problem
    return scenario_creator(sname, **sca)

######## end helper functions #########

# special helper function
def id_fix_list_fct(s):
    """ specify tuples used by the classic (non-RC-based) fixer.

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
    data_dir = "data" + os.sep + instname + os.sep + "scenariodata"
    if not os.path.isdir(data_dir):
        print(msg, "\n   bad instance name=", instname)
        quit()
    try:
        bunper = int(sys.argv[2])
    except Exception:
        print(msg, "\n    bad number of bundles per rank=", sys.argv[2])
        quit()
    try:
        maxit = int(sys.argv[3])
    except Exception:
        print(msg, "\n    bad max iterations=", sys.argv[3])
        quit()
    try:
        rho = int(sys.argv[4])
    except Exception:
        print(msg, "\n    bad rho=", sys.argv[4])
        quit()

    # The number of scenarios is the last number in the instance name
    ScenCount = sputils.extract_num(instname)

    start_time = dt.datetime.now()

    options = {}
    options["solver_name"] = "gurobi_persistent"
    options["PHIterLimit"] = maxit
    options["defaultPHrho"] = rho
    options["convthresh"] = -1
    options["subsolvedirectives"] = None
    options["verbose"] = False
    options["display_timing"] = False
    options["display_progress"] = True

    options["primal_dual_converger_options"] = {"tol" : 1e-6}
    options["display_convergence_detail"] = True
    options["smoothed"] = True
    options["defaultPHp"] = .5
    options["defaultPHbeta"] = 0.1

    ### async section ###
    options["asynchronous"] = False
    options["async_frac_needed"] = 0.5
    options["async_sleep_secs"] = 1
    ### end asyn section ###
    # one way to set up sub-problem solver options
    options["iter0_solver_options"] = {"mipgap": 0.01}
    # another way
    options["iterk_solver_options"] = {"mipgap": 0.02, "threads": 4}
    options["xhat_solver_options"] = options["iterk_solver_options"]
    if bunper > 0:
        options["bundles_per_rank"] = bunper
    options["append_file_name"] = "sslp.app"

    fixoptions = {}
    fixoptions["verbose"] = True
    fixoptions["boundtol"] = 0.01
    fixoptions["id_fix_list_fct"] = id_fix_list_fct

    options["fixeroptions"] = fixoptions

    options["gapperoptions"] = {
        "verbose": True,
        "mipgapdict": {0: 0.02, 1: 0.02, 5: 0.01, 10: 0.005},
    }

    all_scenario_names = list()
    for sn in range(ScenCount):
        all_scenario_names.append("Scenario" + str(sn + 1))

    ph = mpisppy.opt.ph.PH(
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement,
        scenario_creator_kwargs={"data_dir": data_dir},
        ph_converger = PrimalDualConverger
    )

    if ph.cylinder_rank == 0:
        appfile = options["append_file_name"]
        if not os.path.isfile(appfile):
            with open(appfile, "w") as f:
                f.write("datetime, hostname, instname, solver, n_proc")
                f.write(", bunperank, PHIterLimit, convthresh, Rho")
                f.write(", xhatobj, bound, trivialbnd, lastiter, wallclock")
                f.write(", asyncfrac, asyncsleep")
        if "bundles_per_rank" in options:
            nbunstr = str(options["bundles_per_rank"])
        else:
            nbunstr = "0"
        with open(appfile, "a") as f:
            f.write(
                "\n" + str(start_time) + "," + socket.gethostname() + "," + instname
            )
            f.write(", " + str(options["solver_name"]))
            f.write(", " + str(ph.n_proc))
            f.write(", " + nbunstr)
            f.write(", " + str(options["PHIterLimit"]))
            f.write(", " + str(options["convthresh"]))
            f.write(", " + str(options["defaultPHrho"]))
    ###from mpisppy.xhatlooper import XhatLooper
    conv, eobj, tbound = ph.ph_main()

    # extensions=XhatLooper,

    print("\nQUITTING EARLY; this needs to be a hub and have spokes!!!")
    quit()

    dopts = {"mipgap": 0.001}
    ph.options["asynchronous"] = False
    objbound = ph.post_solve_bound(solver_options=dopts, verbose=False)
    if ph.cylinder_rank == 0:
        print("**** Lagrangian objective function bound=", objbound)

    end_time = dt.datetime.now()

    if ph.cylinder_rank == 0:
        with open(appfile, "a") as f:
            f.write(", " + str(objbound) + ", " + str(tbound) + ", " + str(ph._PHIter))
            f.write(", " + str((end_time - start_time).total_seconds()))
            if options["asynchronous"]:
                f.write(", " + str(options["async_frac_needed"]))
                f.write(", " + str(options["async_sleep_secs"]))
