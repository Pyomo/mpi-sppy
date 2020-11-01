# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# general example driver for farmer with cylinders

import mpisppy.examples.uc.uc_funcs as uc

# Make it all go
from mpisppy.utils.sputils import spin_the_wheel
from mpisppy.examples import baseparsers
from mpisppy.examples import vanilla
from mpisppy.cylinders.hub import LShapedHub
from mpisppy.opt.lshaped import LShapedMethod


def _parse_args():
    parser = baseparsers.make_parser(num_scens_reqd=False)
    parser = baseparsers.two_sided_args(parser)
    parser = baseparsers.fwph_args(parser)
    parser = baseparsers.xhatlshaped_args(parser)
    args = parser.parse_args()
    # Need default_rho for FWPH, without you get 
    # uninitialized numeric value error
    if args.with_fwph and args.default_rho is None:
        print("Must specify a default_rho if using FWPH")
        quit()

    return args


def main():
    args = _parse_args()

    num_scen = args.num_scens
    with_fwph = args.with_fwph
    with_xhatlshaped = args.with_xhatlshaped

    scenario_creator = uc.scenario_creator
    scenario_denouement = uc.scenario_denouement
    cb_data = {"path": f"./{num_scen}scenarios_r1/"}
    all_scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]

    # Things needed for vanilla cylinders
    beans = (args, scenario_creator, scenario_denouement, all_scenario_names)

    # Options for the L-shaped method at the hub
    spo = None if args.max_solver_threads is None else {"threads": args.max_solver_threads}
    spo['mipgap'] = 0.005
    options = {
        "master_solver": args.solver_name,
        "sp_solver": args.solver_name,
        "sp_solver_options" : spo,
        "master_solver_options" : spo,
        #"valid_eta_lb": {n:0. for n in all_scenario_names},
        "max_iter": args.max_iterations,
        "verbose": False,
        "master_scenarios":[all_scenario_names[len(all_scenario_names)//2]],
   }
    
    # L-shaped hub
    hub_dict = {
        "hub_class": LShapedHub,
        "hub_kwargs": {
            "options": {
                "rel_gap": args.rel_gap,
                "abs_gap": args.abs_gap,
            },
        },
        "opt_class": LShapedMethod,
        "opt_kwargs": { # Args passed to LShapedMethod __init__
            "options": options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
        },
    }

    # FWPH spoke
    if with_fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, cb_data=cb_data)
        fw_spoke["opt_kwargs"]["PH_options"]["abs_gap"] = 0.
        fw_spoke["opt_kwargs"]["PH_options"]["rel_gap"] = 1e-5
        fw_spoke["opt_kwargs"]["rho_setter"] = uc.scenario_rhos

    if with_xhatlshaped:
        xhatlshaped_spoke = vanilla.xhatlshaped_spoke(*beans, cb_data=cb_data)

    list_of_spoke_dict = list()
    if with_fwph:
        list_of_spoke_dict.append(fw_spoke)
    if with_xhatlshaped:
        list_of_spoke_dict.append(xhatlshaped_spoke)

    spin_the_wheel(hub_dict, list_of_spoke_dict)


if __name__ == "__main__":
    main()
