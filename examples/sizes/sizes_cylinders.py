# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import sys
import copy
import sizes

from mpisppy.utils.sputils import spin_the_wheel
from mpisppy.extensions.fixer import Fixer
from mpisppy.utils import baseparsers
from mpisppy.utils import vanilla


def _parse_args():
    parser = baseparsers.make_parser()
    parser = baseparsers.two_sided_args(parser)
    parser = baseparsers.mip_options(parser)
    parser = baseparsers.fixer_args(parser)
    parser = baseparsers.fwph_args(parser)
    parser = baseparsers.lagrangian_args(parser)
    parser = baseparsers.xhatlooper_args(parser)
    parser = baseparsers.xhatshuffle_args(parser)
    args = parser.parse_args()
    return args


def main():
    
    args = _parse_args()

    num_scen = args.num_scens

    with_fwph = args.with_fwph
    with_xhatlooper = args.with_xhatlooper
    with_xhatshuffle = args.with_xhatshuffle
    with_lagrangian = args.with_lagrangian
    with_fixer = args.with_fixer
    fixer_tol = args.fixer_tol

    if num_scen not in (3, 10):
        raise RuntimeError(f"num_scen must the 3 or 10; was {num_scen}")
    
    scenario_creator_kwargs = {"scenario_count": num_scen}
    scenario_creator = sizes.scenario_creator
    scenario_denouement = sizes.scenario_denouement
    all_scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]
    rho_setter = sizes._rho_setter
    
    if with_fixer:
        ph_ext = Fixer
    else:
        ph_ext = None

    # Things needed for vanilla cylinders
    beans = (args, scenario_creator, scenario_denouement, all_scenario_names)        
    # Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              scenario_creator_kwargs=scenario_creator_kwargs,
                              ph_extensions=ph_ext,
                              rho_setter = rho_setter)

    if with_fixer:
        hub_dict["opt_kwargs"]["options"]["fixeroptions"] = {
            "verbose": False,
            "boundtol": fixer_tol,
            "id_fix_list_fct": sizes.id_fix_list_fct,
        }
    if args.default_rho is None:
        # since we are using a rho_setter anyway
        hub_dict.opt_kwargs.options["defaultPHrho"] = 1  
    
    # FWPH spoke
    if with_fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # Standard Lagrangian bound spoke
    if with_lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(
            *beans,
            scenario_creator_kwargs=scenario_creator_kwargs,
            rho_setter=rho_setter,
        )

    # xhat looper bound spoke
    if with_xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(
            *beans,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )

    # xhat shuffle bound spoke
    if with_xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(
            *beans,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )
       
    list_of_spoke_dict = list()
    if with_fwph:
        list_of_spoke_dict.append(fw_spoke)
    if with_lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if with_xhatlooper:
        list_of_spoke_dict.append(xhatlooper_spoke)
    if with_xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    spin_the_wheel(hub_dict, list_of_spoke_dict)


if __name__ == "__main__":
    main()
