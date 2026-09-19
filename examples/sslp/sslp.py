###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# NOTE: This example used loose bundles, which are no longer supported. See doc/src/properbundles.rst for proper bundles.
# update April 2020: BUT this really needs upper and lower bound spokes
# dlw February 2019: PySP 2 for the sslp example

import os
import mpisppy.scenario_tree as scenario_tree
import mpisppy.extensions.fixer as fixer

import model.ReferenceModel as ref


def scenario_creator(scenario_name, data_dir=None, surrogate=False, cfg=None):
    """ The callback needs to create an instance and then attach
        the PySP nodes to it in a list _mpisppy_node_list ordered by stages.
        Optionally attach _PHrho.
        cfg is just there to keep callers (e.g. bundler) happy.
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
def scenario_names_creator(num_scens, start=None):
    # one-based scenario labels (Scenario1, Scenario2, ...) -- the on-disk
    # data dir is 1-indexed (Scenario1.dat ... ScenarioN.dat).
    # `start` follows the mpi-sppy convention used by farmer/aircond/uc:
    # 0-based offset = count of already-used scenarios. proper_bundler
    # calls this with start=firstnum-inum (== 0 for the first bundle),
    # so the offset must be 0-based even though the *labels* are 1-based.
    if start is None:
        start = 0
    return [f"Scenario{i+1}" for i in range(start, start+num_scens)]


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
    cfg.add_to_config("surrogate_nonant",
                      description="use a surrogate nonant summing the total number of servers",
                      domain=bool,
                      default=False,
                     )


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
    kwargs = {"data_dir": data_dir, "surrogate": cfg.surrogate_nonant}
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
