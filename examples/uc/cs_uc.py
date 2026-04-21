###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# UC cross-scenario-cuts demonstration.
# Hub: PH + CrossScenarioExtension (+ Gapper, optional Fixer)
# Spokes: XhatLooperInnerBound, CrossScenarioCutSpoke
#
# Proper bundles (``--scenarios-per-bundle``) are supported, but because the
# UC rho setter and fixer read per-scenario attributes (TimePeriods,
# ThermalGenerators) and because the cross-scenario-cut spoke expects
# individual-scenario subproblems, enabling bundles here automatically:
#   - disables the cross-scenario-cut spoke and extension
#   - drops uc._rho_setter and uc.id_fix_list_fct
# i.e., with bundles this script reduces to a plain PH + xhat looper run.

import json
import sys

import uc_funcs as uc

from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils import config, proper_bundler
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.extensions.extension import MultiExtension
from mpisppy.extensions.fixer import Fixer
from mpisppy.extensions.mipgapper import Gapper
from mpisppy.extensions.cross_scen_extension import CrossScenarioExtension


def _parse_args():
    cfg = config.Config()
    cfg.popular_args()
    cfg.num_scens_required()
    cfg.ph_args()
    cfg.two_sided_args()
    cfg.fixer_args()
    cfg.xhatlooper_args()
    cfg.cross_scenario_cuts_args()
    cfg.proper_bundle_config()
    cfg.add_to_config("ph_mipgaps_json",
                      description="json file with mipgap schedule (default None)",
                      domain=str,
                      default=None)
    cfg.parse_command_line("cs_uc")
    # cs_uc's identity is cross-scenario cuts; default it on, but honor an
    # explicit --no-cross-scenario-cuts. (Pyomo's bool flag default of False
    # is indistinguishable from "user passed --no-...", so check argv.)
    if "--no-cross-scenario-cuts" not in sys.argv:
        cfg.cross_scenario_cuts = True
    return cfg


def _setup_bundles(cfg):
    """Return (scenario_creator, kwargs, scenario_names, bundle_wrapper)."""
    cfg.quick_assign("turn_off_names_check", bool, True)
    bundle_wrapper = proper_bundler.ProperBundler(uc)
    bundle_wrapper.set_bunBFs(cfg)
    scenario_creator = bundle_wrapper.scenario_creator
    scenario_creator_kwargs = bundle_wrapper.kw_creator(cfg)
    num_buns = cfg.num_scens // cfg.scenarios_per_bundle
    all_scenario_names = bundle_wrapper.bundle_names_creator(num_buns, cfg=cfg)
    return (scenario_creator, scenario_creator_kwargs, all_scenario_names,
            bundle_wrapper)


def main():
    cfg = _parse_args()

    bundling = cfg.get("scenarios_per_bundle") is not None
    if bundling and cfg.cross_scenario_cuts:
        print("cs_uc.py: --scenarios-per-bundle requested; disabling "
              "cross-scenario cuts (see module docstring).", file=sys.stderr)
        cfg.cross_scenario_cuts = False

    if bundling:
        (scenario_creator, scenario_creator_kwargs, all_scenario_names,
         bundle_wrapper) = _setup_bundles(cfg)
        rho_setter = None
        id_fix_list_fct = None
        _raw_denouement = uc.scenario_denouement
        def scenario_denouement(rank, sname, s):
            if "Bundle" not in sname:
                _raw_denouement(rank, sname, s)
    else:
        scenario_creator = uc.scenario_creator
        scenario_creator_kwargs = uc.kw_creator(cfg)
        all_scenario_names = uc.scenario_names_creator(cfg.num_scens)
        scenario_denouement = uc.scenario_denouement
        rho_setter = uc._rho_setter
        id_fix_list_fct = uc.id_fix_list_fct

    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    hub_dict = vanilla.ph_hub(*beans,
                              scenario_creator_kwargs=scenario_creator_kwargs,
                              ph_extensions=MultiExtension,
                              rho_setter=rho_setter)

    ext_classes = []
    if cfg.cross_scenario_cuts:
        ext_classes.append(CrossScenarioExtension)
    if cfg.fixer and id_fix_list_fct is not None:
        ext_classes.append(Fixer)
    if cfg.ph_mipgaps_json is not None:
        ext_classes.append(Gapper)
        with open(cfg.ph_mipgaps_json) as fin:
            din = json.load(fin)
        mipgapdict = {int(i): din[i] for i in din}
        hub_dict["opt_kwargs"]["options"]["gapperoptions"] = {
            "verbose": cfg.verbose,
            "mipgapdict": mipgapdict,
        }
    hub_dict["opt_kwargs"]["extension_kwargs"] = {"ext_classes": ext_classes}

    if cfg.cross_scenario_cuts:
        hub_dict["opt_kwargs"]["options"]["cross_scen_options"] = {
            "check_bound_improve_iterations": cfg.cross_scenario_iter_cnt,
        }

    if cfg.fixer and id_fix_list_fct is not None:
        hub_dict["opt_kwargs"]["options"]["fixeroptions"] = {
            "verbose": cfg.verbose,
            "boundtol": cfg.fixer_tol,
            "id_fix_list_fct": id_fix_list_fct,
        }

    xhatlooper_spoke = vanilla.xhatlooper_spoke(
        *beans, scenario_creator_kwargs=scenario_creator_kwargs)

    list_of_spoke_dict = [xhatlooper_spoke]
    if cfg.cross_scenario_cuts:
        # The cross-scen spoke uses the raw per-scenario creator even when the
        # hub is bundled, but we've already disabled cuts in that case.
        cross_spoke = vanilla.cross_scenario_cuts_spoke(
            *beans, scenario_creator_kwargs=scenario_creator_kwargs)
        list_of_spoke_dict.append(cross_spoke)

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    if wheel.global_rank == 0:
        print(f"BestInnerBound={wheel.BestInnerBound} and "
              f"BestOuterBound={wheel.BestOuterBound}")


if __name__ == "__main__":
    main()
