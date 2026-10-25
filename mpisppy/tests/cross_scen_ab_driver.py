###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""One leg of the cross-scenario cuts checkpoint test, as its own MPI job.

Run under mpiexec by ``test_checkpoint_cylinders.py``::

  mpiexec -np 2 python -m mpi4py mpisppy/tests/cross_scen_ab_driver.py \\
      --out /tmp/leg.json --num-scens 3 --solver-name ... --cross-scenario-cuts

A PH hub with CrossScenarioExtension and the cross-scenario cut spoke, on
farmer. generic_cylinders does not build this pair, so this driver does it
with cfg_vanilla, the way examples/netdes/netdes_cylinders.py does.

The hub rank writes the snapshot that ``cylinders_ab_driver`` writes, plus
the cuts on each scenario model and the extension's state as it was restored.

Not named ``test_*`` on purpose: it is a helper, and pytest must not collect
it.
"""

import json
import sys
import time

from pyomo.core.expr.visitor import identify_variables

import mpisppy.MPI as MPI
import mpisppy.tests.examples.farmer as farmer
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.config as config
from mpisppy.extensions.cross_scen_extension import CrossScenarioExtension
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.cylinders_ab_driver import _hub_snapshot

#: The extension's state right after restore_state, or None if it was not
#: restored. Recorded then because the run moves it on afterwards.
_restored_state = {}
#: The cut keys on each scenario model when the hub first reads the spoke's
#: cuts. On a resume that is before any new cut can arrive, so it shows what
#: post_iter0 left, apart from a cut the restarted spoke sends again.
_keys_at_first_read = {}
#: The cut keys and the extension state when the last checkpoint took the
#: extension's state. Cuts that arrive after it are not in the checkpoint.
_at_last_checkpoint = {}


def _recording_restore(real_restore):
    def restore(self, state):
        real_restore(self, state)
        _restored_state.update(self.checkpoint_state())
    return restore


def _recording_checkpoint_state(real_state):
    def state(self):
        result = real_state(self)
        _at_last_checkpoint["state"] = dict(result)
        _at_last_checkpoint["cut_keys"] = {
            sname: sorted(str(k) for k in s._mpisppy_model.benders_cuts.keys())
            for sname, s in self.opt.local_scenarios.items()}
        return result
    return state


def _waiting_enditer(real_enditer, iteration, timeout=600.):
    """At ``iteration``, keep the hub talking to the cut spoke until every
    hub rank holds a cut this run received (on a resume, one keyed past the
    restored cuts). How soon cuts arrive depends on the machine; on CI the
    first one can come after the stopped leg's last checkpoint, or after a
    resumed leg's last iteration."""
    def enditer(self):
        real_enditer(self)
        if self.opt._PHIter != iteration:
            return
        start = time.time()
        while True:
            self.opt.spcomm.sync_nonants()
            self.sync_with_spokes()
            have = int(all(
                any(key[0] >= self._cut_key_offset
                    for key in s._mpisppy_model.benders_cuts)
                for s in self.opt.local_scenarios.values()))
            if self.opt.mpicomm.allreduce(have, op=MPI.MIN):
                return
            if time.time() - start > timeout:
                raise RuntimeError(f"no cut arrived in {timeout} seconds")
            time.sleep(0.01)
    return enditer


def _recording_first_read(real_read):
    def read(self):
        if not _keys_at_first_read:
            for sname, s in self.opt.local_scenarios.items():
                _keys_at_first_read[sname] = sorted(
                    str(k) for k in s._mpisppy_model.benders_cuts.keys())
        return real_read(self)
    return read


def _extension(opt):
    candidates = list(getattr(opt.extobject, "extdict", {}).values())
    candidates.append(opt.extobject)
    return next(e for e in candidates if isinstance(e, CrossScenarioExtension))


def _cuts(opt):
    """Each scenario's cuts by key, as strings JSON can hold, and whether
    every eta in them is the model's current eta component."""
    cuts, current_eta = {}, True
    for sname, s in opt.local_scenarios.items():
        model = s._mpisppy_model
        cuts[sname] = {
            "benders_cuts": {str(k): str(c.expr)
                             for k, c in model.benders_cuts.items()},
            "inner_bound_constr": {str(k): str(c.expr)
                                   for k, c in model.inner_bound_constr.items()},
        }
        for c in model.benders_cuts.values():
            for v in identify_variables(c.body):
                if v.parent_component().local_name == "eta" \
                        and v.parent_component() is not model.eta:
                    current_eta = False
    return cuts, current_eta


def main():
    if sys.argv[1] != "--out":
        raise RuntimeError("usage: cross_scen_ab_driver.py --out PATH [args]")
    out_path = sys.argv[2]
    args = sys.argv[3:]
    wait_at = None
    if "--wait-for-cuts-at" in args:
        i = args.index("--wait-for-cuts-at")
        wait_at = int(args[i + 1])
        del args[i:i + 2]
    sys.argv = [sys.argv[0]] + args

    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.cross_scenario_cuts_args()
    cfg.checkpoint_args()
    cfg.parse_command_line("cross_scen_ab_driver")

    CrossScenarioExtension.restore_state = _recording_restore(
        CrossScenarioExtension.restore_state)
    CrossScenarioExtension.get_from_cross_cuts = _recording_first_read(
        CrossScenarioExtension.get_from_cross_cuts)
    CrossScenarioExtension.checkpoint_state = _recording_checkpoint_state(
        CrossScenarioExtension.checkpoint_state)
    if wait_at is not None:
        CrossScenarioExtension.enditer = _waiting_enditer(
            CrossScenarioExtension.enditer, wait_at)

    names = farmer.scenario_names_creator(cfg.num_scens)
    kwargs = {"use_integer": False, "crops_multiplier": 1,
              "num_scens": cfg.num_scens}
    beans = (cfg, farmer.scenario_creator, farmer.scenario_denouement, names)
    hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=kwargs)
    vanilla.add_cross_scenario_cuts(hub_dict, cfg)
    spoke_dict = vanilla.cross_scenario_cuts_spoke(
        *beans, scenario_creator_kwargs=kwargs)
    wheel = WheelSpinner(hub_dict, [spoke_dict])
    wheel.spin()

    if wheel.on_hub():
        opt = wheel.spcomm.opt
        ext = _extension(opt)
        # Collective over the hub's ranks (Eobjective all-reduces), so every
        # hub rank takes one; rank 0's is the one written.
        snapshot = _hub_snapshot(wheel)
        if wheel.cylinder_rank != 0:
            return
        snapshot["cuts"], snapshot["cuts_use_current_eta"] = _cuts(opt)
        snapshot["cut_key_offset"] = ext._cut_key_offset
        # Taken before the final checkpoint_state call below overwrites it.
        snapshot["at_last_checkpoint"] = dict(_at_last_checkpoint) or None
        snapshot["final_extension_state"] = ext.checkpoint_state()
        snapshot["restored_extension_state"] = _restored_state or None
        snapshot["cut_keys_at_first_read"] = _keys_at_first_read
        with open(out_path, "w") as f:
            json.dump(snapshot, f)


if __name__ == "__main__":
    main()
