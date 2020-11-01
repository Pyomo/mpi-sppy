# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# general example driver for the hydro example with cylinders

import mpisppy.examples.hydro.hydro as hydro

from mpisppy.utils.sputils import spin_the_wheel
from mpisppy.examples import baseparsers
from mpisppy.examples import vanilla

import mpisppy.cylinders as cylinders

# For this problem, the subproblems are
# small and take no time to solve. The
# default SPOKE_SLEEP_TIME of 0.1 *causes*
# synchronization issues in this case, so
# we reduce it so as not to dominate the
# time spent for cylinder synchronization
cylinders.SPOKE_SLEEP_TIME = 0.0001

def _parse_args():
    parser = baseparsers.make_multistage_parser()
    parser = baseparsers.two_sided_args(parser)
    parser = baseparsers.xhatlooper_args(parser)
    parser = baseparsers.xhatshuffle_args(parser)
    parser = baseparsers.lagrangian_args(parser)
    parser = baseparsers.xhatspecific_args(parser)
    args = parser.parse_args()
    return args


def main():

    args = _parse_args()

    BFs = [int(bf) for bf in args.BFs.split(',')]
    if len(BFs) != 2:
        raise RuntimeError("Hydro is a three stage problem, so it needs 2 BFs")

    with_xhatspecific = args.with_xhatspecific
    with_lagrangian = args.with_lagrangian

    # This is multi-stage, so we need to supply node names
    all_nodenames = ["ROOT"] # all trees must have this node
    # The rest is a naming convention invented for this problem.
    # Note that mpisppy does not have nodes at the leaves,
    # and node names must end in a serial number.
    for b in range(BFs[0]):
        all_nodenames.append("ROOT_"+str(b))

    ScenCount = BFs[0] * BFs[1]
    cb_data = BFs
    all_scenario_names = [f"Scen{i+1}" for i in range(ScenCount)]
    scenario_creator = hydro.scenario_creator
    scenario_denouement = hydro.scenario_denouement
    rho_setter = None
    
    # Things needed for vanilla cylinders
    beans = (args, scenario_creator, scenario_denouement, all_scenario_names)
    
    # Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              cb_data=cb_data,
                              ph_extensions=None,
                              rho_setter = rho_setter)
    hub_dict["opt_kwargs"]["all_nodenames"] = all_nodenames
    hub_dict["opt_kwargs"]["PHoptions"]["branching_factors"] = BFs

    # Standard Lagrangian bound spoke
    if with_lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              cb_data=cb_data,
                                              rho_setter = rho_setter)
        lagrangian_spoke["opt_kwargs"]["all_nodenames"] = all_nodenames
        lagrangian_spoke["opt_kwargs"]["PHoptions"]["branching_factors"] = BFs

    # xhat looper bound spoke
    xhat_scenario_dict = {"ROOT": "Scen1",
                          "ROOT_0": "Scen1",
                          "ROOT_1": "Scen4",
                          "ROOT_2": "Scen7"}
    
    if with_xhatspecific:
        xhatspecific_spoke = vanilla.xhatspecific_spoke(*beans,
                                                        xhat_scenario_dict,
                                                        all_nodenames,
                                                        BFs,
                                                        cb_data=cb_data)

    list_of_spoke_dict = list()
    if with_lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if with_xhatspecific:
        list_of_spoke_dict.append(xhatspecific_spoke)

    spcomm, opt_dict = spin_the_wheel(hub_dict, list_of_spoke_dict)

    if "hub_class" in opt_dict:  # we are a hub rank
        if spcomm.opt.rank == spcomm.opt.rank0:  # we are the reporting hub rank
            print("BestInnerBound={} and BestOuterBound={}".\
                  format(spcomm.BestInnerBound, spcomm.BestOuterBound))
    

if __name__ == "__main__":
    main()
