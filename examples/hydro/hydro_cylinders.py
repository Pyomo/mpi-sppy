# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# general example driver for the hydro example with cylinders

import hydro

import mpisppy.utils.sputils as sputils

from mpisppy.utils import baseparsers
from mpisppy.utils import vanilla

import mpisppy.cylinders as cylinders

# For this problem, the subproblems are
# small and take no time to solve. The
# default SPOKE_SLEEP_TIME of 0.1 *causes*
# synchronization issues in this case, so
# we reduce it so as not to dominate the
# time spent for cylinder synchronization
cylinders.SPOKE_SLEEP_TIME = 0.0001

write_solution = True

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

    BFs = args.branching_factors
    if len(BFs) != 2:
        raise RuntimeError("Hydro is a three stage problem, so it needs 2 BFs")

    with_xhatshuffle = args.with_xhatshuffle
    with_lagrangian = args.with_lagrangian

    # This is multi-stage, so we need to supply node names
    all_nodenames = sputils.create_nodenames_from_BFs(BFs)

    ScenCount = BFs[0] * BFs[1]
    scenario_creator_kwargs = {"branching_factors": BFs}
    all_scenario_names = [f"Scen{i+1}" for i in range(ScenCount)]
    scenario_creator = hydro.scenario_creator
    scenario_denouement = hydro.scenario_denouement
    rho_setter = None
    
    # Things needed for vanilla cylinders
    beans = (args, scenario_creator, scenario_denouement, all_scenario_names)
    
    # Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              scenario_creator_kwargs=scenario_creator_kwargs,
                              ph_extensions=None,
                              rho_setter = rho_setter,
                              all_nodenames = all_nodenames)

    # Standard Lagrangian bound spoke
    if with_lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = rho_setter,
                                              all_nodenames = all_nodenames)

    # xhat looper bound spoke
    
    if with_xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans,
                                                        all_nodenames=all_nodenames,
                                                        scenario_creator_kwargs=scenario_creator_kwargs)

    list_of_spoke_dict = list()
    if with_lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if with_xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    spcomm, opt_dict = sputils.spin_the_wheel(hub_dict, list_of_spoke_dict)

    if "hub_class" in opt_dict:  # we are a hub rank
        if spcomm.opt.cylinder_rank == 0:  # we are the reporting hub rank
            print("BestInnerBound={} and BestOuterBound={}".\
                  format(spcomm.BestInnerBound, spcomm.BestOuterBound))
    
    if write_solution:
        sputils.write_spin_the_wheel_first_stage_solution(spcomm, opt_dict, 'hydro_first_stage.csv')
        sputils.write_spin_the_wheel_tree_solution(spcomm, opt_dict, 'hydro_full_solution')

if __name__ == "__main__":
    main()
