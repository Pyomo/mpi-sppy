# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# general example driver for the hydro example with cylinders

import hydro

import mpisppy.utils.sputils as sputils

from mpisppy.utils import baseparsers
from mpisppy.utils import vanilla
from mpisppy.utils.pysp_model import PySPModel

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

    with_xhatshuffle = args.with_xhatshuffle
    with_lagrangian = args.with_lagrangian

    # This is multi-stage, so we need to supply node names
    hydro = PySPModel("./PySP/models/", "./PySP/nodedata/")
    rho_setter = None
    
    # Things needed for vanilla cylinders
    beans = (args, hydro.scenario_creator, hydro.scenario_denouement, hydro.all_scenario_names)
    
    # Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              ph_extensions=None,
                              rho_setter = rho_setter,
                              all_nodenames=hydro.all_nodenames)

    # Standard Lagrangian bound spoke
    if with_lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              rho_setter = rho_setter,
                                              all_nodenames=hydro.all_nodenames)

    if with_xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans,
                                                      hydro.all_nodenames)

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

    hydro.close()

if __name__ == "__main__":
    main()
