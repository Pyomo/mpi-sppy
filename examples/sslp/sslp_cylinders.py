# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import sys
import os
import copy
import mpisppy.examples.sslp.sslp as sslp

from mpisppy.utils.sputils import spin_the_wheel
from mpisppy.extensions.fixer import Fixer
from mpisppy.examples import baseparsers
from mpisppy.examples import vanilla


def _parse_args():
    parser = baseparsers.make_parser(num_scens_reqd=False)
    parser.add_argument("--instance-name",
                        help="sslp instance name (e.g., sslp_15_45_10)",
                        dest="instance_name",
                        type=str,
                        default=None)                
    parser = baseparsers.two_sided_args(parser)
    parser = baseparsers.fixer_args(parser)
    parser = baseparsers.xhatlooper_args(parser)
    parser = baseparsers.fwph_args(parser)
    parser = baseparsers.lagrangian_args(parser)
    parser = baseparsers.xhatshuffle_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    inst = args.instance_name
    num_scen = int(inst.split("_")[-1])
    if args.num_scens is not None and args.num_scens != num_scen:
        raise RuntimeError("Argument num-scens={} does not match the number "
                           "implied by instance name={} "
                           "\n(--num-scens is not needed for sslp)")

    with_fwph = args.with_fwph
    with_fixer = args.with_fixer
    fixer_tol = args.fixer_tol
    with_xhatlooper = args.with_xhatlooper
    with_xhatshuffle = args.with_xhatshuffle
    with_lagrangian = args.with_lagrangian

    if args.default_rho is None:
        raise RuntimeError("The --default-rho option must be specified")

    cb_data = f"{sslp.__file__[:-8]}/data/{inst}/scenariodata"
    scenario_creator = sslp.scenario_creator
    scenario_denouement = sslp.scenario_denouement    
    all_scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]

    if with_fixer:
        ph_ext = Fixer
    else:
        ph_ext = None

    # Things needed for vanilla cylinders
    beans = (args, scenario_creator, scenario_denouement, all_scenario_names)

    # Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              cb_data=cb_data,
                              ph_extensions=ph_ext,
                              rho_setter = None)

    if with_fixer:
        hub_dict["opt_kwargs"]["PHoptions"]["fixeroptions"] = {
            "verbose": False,
            "boundtol": fixer_tol,
            "id_fix_list_fct": sslp.id_fix_list_fct,
        }
        
    # FWPH spoke
    if with_fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, cb_data=cb_data)

    # Standard Lagrangian bound spoke
    if with_lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              cb_data=cb_data,
                                              rho_setter = None)
        
    # xhat looper bound spoke
    if with_xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(*beans, cb_data=cb_data)

    # xhat shuffle bound spoke
    if with_xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, cb_data=cb_data)
       
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
