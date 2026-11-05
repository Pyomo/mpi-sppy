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

Hub rank 0 writes the snapshot that ``cylinders_ab_driver`` writes, plus the
cuts on each scenario model and the extension's state as it was restored, with
what each hub rank holds gathered into it.

Not named ``test_*`` on purpose: it is a helper, and pytest must not collect
it.
"""

import json
import sys
import time

import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables

import mpisppy.MPI as MPI
import mpisppy.tests.examples.farmer as farmer
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.config as config
from mpisppy.extensions.cross_scen_extension import CrossScenarioExtension
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.cylinders_ab_driver import _hub_snapshot
from mpisppy.utils.sputils import find_active_objective, is_persistent

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
#: What each outer bound check in this leg computed, before the hub's best
#: bound absorbs it. Checks happen only after Iter0, so on a resume only after
#: the restore.
_checked_bounds = []


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


def _recording_check_bound(real_check_bound):
    def check_bound(self):
        spcomm = self.opt.spcomm
        real_update = spcomm.OuterBoundUpdate

        def update(bound, *args, **kwargs):
            _checked_bounds.append(float(bound))
            return real_update(bound, *args, **kwargs)

        spcomm.OuterBoundUpdate = update
        try:
            return real_check_bound(self)
        finally:
            del spcomm.OuterBoundUpdate
    return check_bound


def _waiting_enditer(real_enditer, iteration, timeout=600.):
    """At ``iteration``, keep the hub talking to the cut spoke until every
    hub rank holds a cut this run received (on a resume, one keyed past the
    restored cuts). How soon cuts arrive depends on the machine; on CI the
    first one can come after the stopped leg's last checkpoint, or too late
    for a resumed leg to check the bound with it."""
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


def _solve_on_persistent(plugin, objective):
    plugin.set_objective(objective)
    results = plugin.solve(tee=False, load_solutions=False)
    condition = results.solver.termination_condition
    if condition != pyo.TerminationCondition.optimal:
        return str(condition)
    plugin.load_vars()
    return pyo.value(objective)


def _solve_from_the_model(s, solver, objective):
    results = solver.solve(s, tee=False, load_solutions=False)
    condition = results.solver.termination_condition
    if condition != pyo.TerminationCondition.optimal:
        return str(condition)
    s.solutions.load_from(results)
    return pyo.value(objective)


def _solver_vs_model(opt, solver_name):
    """What each scenario's persistent solver answers, beside what a solver
    built now from the Pyomo model answers, or None without persistent solvers.

    The cuts and eta bounds on the Pyomo model say nothing about the persistent
    solver, which the bound checks and PH solves use. It is given them
    separately, or on a resume inherits them, so it can fall out of step. The
    problems are the EF objective, which the cuts move, and each eta alone,
    which its bound stops. Asking the solvers rather than reading a count from
    one keeps this to one code path on every solver CI might pick.

    The inner-bound rows come out first. Each says EF_obj is at least the last
    checked bound, and on a resumed leg that row sets the EF objective's
    optimum whether or not the cuts are there.

    Only for the end of a leg: this changes the objectives and removes those
    rows.
    """
    if not all(is_persistent(s._solver_plugin)
               for s in opt.local_scenarios.values()):
        return None
    direct = pyo.SolverFactory(solver_name.replace("_persistent", "_direct"))
    answers = {}
    for sname, s in opt.local_scenarios.items():
        model = s._mpisppy_model
        for row in model.inner_bound_constr.values():
            s._solver_plugin.remove_constraint(row)
            row.deactivate()
        find_active_objective(s).deactivate()
        model.EF_Obj.activate()
        answers[sname] = {"EF_Obj": [
            _solve_on_persistent(s._solver_plugin, model.EF_Obj),
            _solve_from_the_model(s, direct, model.EF_Obj)]}
        model.EF_Obj.deactivate()
        for eta_name in opt.all_scenario_names:
            probe = pyo.Objective(expr=model.eta[eta_name])
            model.add_component("_eta_probe", probe)
            answers[sname][f"eta[{eta_name}]"] = [
                _solve_on_persistent(s._solver_plugin, probe),
                _solve_from_the_model(s, direct, probe)]
            model.del_component(probe)
    return answers


def _cuts_the_solver_lacks(opt):
    """The model's cuts each persistent solver has not been given, by the
    plugin's own record. A cut that does not bind leaves every answer in
    _solver_vs_model the same, so those answers cannot show that one is
    missing. Call this before _solver_vs_model, which removes rows."""
    lacking = {}
    for sname, s in opt.local_scenarios.items():
        if not is_persistent(s._solver_plugin):
            continue
        held = s._solver_plugin._pyomo_con_to_solver_con_map
        model = s._mpisppy_model
        lacking[sname] = [
            c.name for c in (*model.benders_cuts.values(),
                             *model.inner_bound_constr.values())
            if c not in held]
    return lacking


def _merged(per_rank):
    """One dict from each hub rank's dict, which have different scenarios."""
    return {k: v for d in per_rank for k, v in d.items()}


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
    maximize = "--maximize" in args
    if maximize:
        args.remove("--maximize")
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
    CrossScenarioExtension._check_bound = _recording_check_bound(
        CrossScenarioExtension._check_bound)
    if wait_at is not None:
        CrossScenarioExtension.enditer = _waiting_enditer(
            CrossScenarioExtension.enditer, wait_at)

    names = farmer.scenario_names_creator(cfg.num_scens)
    kwargs = {"use_integer": False, "crops_multiplier": 1,
              "num_scens": cfg.num_scens,
              "sense": pyo.maximize if maximize else pyo.minimize}
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
        # hub rank takes one. Rank 0's scalars are the ones written.
        snapshot = _hub_snapshot(wheel)
        cuts, current_eta = _cuts(opt)
        # Taken before the final checkpoint_state call below overwrites it.
        at_last_checkpoint = dict(_at_last_checkpoint)
        hub_ranks = opt.mpicomm.gather({
            "hub_state": snapshot["state"],
            "cuts": cuts,
            "current_eta": current_eta,
            "cut_key_offset": ext._cut_key_offset,
            "at_last_checkpoint": at_last_checkpoint,
            "final_extension_state": ext.checkpoint_state(),
            "first_read": _keys_at_first_read,
            "checked_bounds": _checked_bounds,
            "restored_extension_state": _restored_state or None,
            "cuts_the_solver_lacks": _cuts_the_solver_lacks(opt),
            "solver_vs_model": _solver_vs_model(opt, cfg.solver_name),
        }, root=0)
        if wheel.cylinder_rank != 0:
            return
        snapshot["state"] = _merged(r["hub_state"] for r in hub_ranks)
        snapshot["cuts"] = _merged(r["cuts"] for r in hub_ranks)
        snapshot["cuts_use_current_eta"] = all(
            r["current_eta"] for r in hub_ranks)
        # By rank, since they have to agree across the hub's ranks: they
        # decide whether the collective bound check runs.
        snapshot["cut_key_offsets"] = [r["cut_key_offset"] for r in hub_ranks]
        snapshot["final_extension_states"] = [
            r["final_extension_state"] for r in hub_ranks]
        snapshot["checked_bounds"] = [r["checked_bounds"] for r in hub_ranks]
        snapshot["trivial_bound"] = opt.trivial_bound
        snapshot["at_last_checkpoint"] = {
            "states": [r["at_last_checkpoint"].get("state") for r in hub_ranks],
            "cut_keys": _merged(r["at_last_checkpoint"].get("cut_keys", {})
                                for r in hub_ranks),
        }
        snapshot["restored_extension_states"] = [
            r["restored_extension_state"] for r in hub_ranks]
        snapshot["cut_keys_at_first_read"] = _merged(
            r["first_read"] for r in hub_ranks)
        snapshot["cuts_the_solver_lacks"] = _merged(
            r["cuts_the_solver_lacks"] for r in hub_ranks)
        snapshot["solver_vs_model"] = (
            None if hub_ranks[0]["solver_vs_model"] is None
            else _merged(r["solver_vs_model"] for r in hub_ranks))
        with open(out_path, "w") as f:
            json.dump(snapshot, f)


if __name__ == "__main__":
    main()
