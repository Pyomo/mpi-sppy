###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""One leg of the cylinders A/B checkpoint harness, as its own MPI job.

Run under mpiexec by ``test_checkpoint_cylinders.py``::

  mpiexec -np 3 python -m mpi4py mpisppy/tests/cylinders_ab_driver.py \\
      --out /tmp/leg.json --module-name farmer --num-scens 3 ...

Everything after ``--out <path>`` is handed to ``generic_cylinders`` untouched,
so each leg is the command line a user would actually type. The hub rank then
writes a JSON snapshot of its final state to ``--out``, which is what the test
compares across legs.

Why a separate process per leg rather than one test that runs three wheels:
the design's acceptance gate calls for the resume to happen in a **fresh
process** (section 11.1), which is also the real use case -- a job stops today
and a new job resumes tomorrow. Spinning a second wheel inside one interpreter
would reuse MPI windows, communicators and whatever module state the first run
left behind, and would prove less.

Not named ``test_*`` on purpose: it is a helper, and pytest must not collect
it.
"""

import json
import sys

from mpisppy import generic_cylinders
from mpisppy.cylinders.hub import PHHub
import mpisppy.utils.checkpointing as ckpt

#: The incumbent a resumed spoke holds right after restoring it, by scenario
#: and variable name. Recorded at the restore because the spoke may improve
#: on it later in the leg, and by then its cache no longer shows what was
#: read from disk.
_restored_values = {}
#: The hub's inner bound at its first convergence check: on a resume, the
#: bound its first resumed iteration runs against, before it has heard
#: anything from a spoke.
_first_inner_bound = {}


def _num(value):
    return None if value is None else float(value)


def _probability_mask(opt):
    """The per-nonant probability coefficients and the zero-probability mask.

    Only ADMM runs make these interesting: the wrapper gives a nonant a
    probability of zero in every subproblem but the one that owns it, and
    mpi-sppy masks W (not prox) accordingly. They are keyed by variable
    *identity* at construction and only their result lands on the model, so
    they are the part of an ADMM resume most likely to come back wrong -- and
    to come back wrong quietly, since a run with a broken mask still solves
    and still reports numbers (design section 8.2, item 3).
    """
    mask = {}
    for sname, s in opt.local_scenarios.items():
        data = s._mpisppy_data
        for ndn, coeffs in getattr(data, "prob_coeff", {}).items():
            values = coeffs.tolist() if hasattr(coeffs, "tolist") \
                else [float(coeffs)]
            mask[f"{sname}|prob_coeff|{ndn}"] = [float(v) for v in values]
        for ndn, m in getattr(data, "prob0_mask", {}).items():
            values = m.tolist() if hasattr(m, "tolist") else [float(m)]
            mask[f"{sname}|prob0_mask|{ndn}"] = [float(v) for v in values]
    return mask


def _fixed_variable_values(opt):
    """Every fixed variable, by name and value.

    ADMM adds dummy vars fixed at 0 after construction; they are not nonants,
    so the nonant-keyed state below never looks at them. A resume that lost
    their fixedness would relax the consensus structure silently.
    """
    import pyomo.environ as pyo
    fixed = {}
    for sname, s in opt.local_scenarios.items():
        for v in s.component_data_objects(pyo.Var):
            if v.is_fixed():
                fixed[f"{sname}|{v.name}"] = _num(v.value)
    return fixed


def _model_holder_is_current(opt):
    """Does whoever built the scenarios point at the models the run iterates?

    None when the scenario_creator is a plain function, which holds nothing.
    False is the ADMM double-memory bug (design section 8.2, item 2): the
    wrapper still referencing the freshly built models a resume replaced, so
    the run carries two copies of every scenario. Nothing else in the run
    reads the wrapper's dictionary, so only a test looking here can see it.
    """
    holder = getattr(getattr(opt, "scenario_creator", None), "__self__", None)
    if holder is None:
        return None
    for attr in ("local_admm_stoch_subproblem_scenarios", "local_scenarios"):
        held = getattr(holder, attr, None)
        if isinstance(held, dict) and held:
            return all(held[sname] is s
                       for sname, s in opt.local_scenarios.items()
                       if sname in held)
    # AdmmBundler holds its bundles as the keys of a dict, not by name.
    held = getattr(holder, "_bundle_varprob", None)
    if isinstance(held, dict):
        current = set(map(id, opt.local_scenarios.values()))
        return all(id(model) in current for model in held)
    return None


def _hub_snapshot(wheel):
    """The hub's final state, in a shape JSON can hold and a test can diff.

    Keys are strings rather than tuples for the same reason: this crosses a
    process boundary. Values are the iterate itself -- nonant values and
    fixedness, and the per-nonant Params that drive the next iteration -- plus
    the scalars a resume is supposed to carry forward.

    One snapshot per hub *rank*: on a multi-rank hub each rank owns a
    different slice of the scenarios, so a comparison that only looked at rank
    0 would leave every other rank's restore unchecked -- which is most of
    what phase 2 added.
    """
    opt = wheel.spcomm.opt
    state = {}
    for sname, s in opt.local_scenarios.items():
        for ndn_i, v in s._mpisppy_data.nonant_indices.items():
            state[f"{sname}|x|{v.name}"] = v._value
            state[f"{sname}|fixed|{v.name}"] = float(v.is_fixed())
            for pname in ("W", "rho", "xbars"):
                param = getattr(s._mpisppy_model, pname, None)
                if param is not None:
                    state[f"{sname}|{pname}|{ndn_i}"] = float(param[ndn_i]._value)

    return {
        "iteration": int(getattr(opt, "_PHIter", 0)),
        "resumed": bool(getattr(opt, "_resumed_from_checkpoint", False)),
        "resume_iteration": int(getattr(opt, "_resume_iteration", 0)),
        "cylinder_rank": int(opt.cylinder_rank),
        "n_proc": int(opt.n_proc),
        "scenario_names": sorted(opt.local_scenarios),
        # Collective (it all-reduces over the cylinder), which is fine because
        # every hub rank builds a snapshot. On a MIP with alternate optima
        # this is the quantity the determinism contract compares to a
        # tolerance, where the per-variable state legitimately differs.
        "objective": _num(opt.Eobjective()),
        "trivial_bound": _num(getattr(opt, "trivial_bound", None)),
        "best_bound_obj_val": _num(getattr(opt, "best_bound_obj_val", None)),
        "best_solution_obj_val": _num(
            getattr(opt, "best_solution_obj_val", None)),
        "BestInnerBound": _num(wheel.BestInnerBound),
        "BestOuterBound": _num(wheel.BestOuterBound),
        # The cylinder credited with that inner bound, which is the one that
        # writes the solution.
        "last_ib_idx": wheel.spcomm.last_ib_idx,
        "first_BestInnerBound": _num(_first_inner_bound.get("value")),
        "state": state,
        # Which Objective each scenario's Eobjective actually reads. A resume
        # rebuilds saved_objectives from the reloaded models, and on a model
        # whose objective was replaced after creation -- CVaR deactivates the
        # risk-neutral one and activates WITH_CVAR -- resolving to the
        # deactivated original would be silently wrong rather than an error.
        "active_objective_names": {
            sname: opt.saved_objectives[sname].name
            for sname in opt.local_scenarios
        },
        "probability_mask": _probability_mask(opt),
        "fixed_variables": _fixed_variable_values(opt),
        "model_holder_is_current": _model_holder_is_current(opt),
    }


def _checkpointer(opt):
    """The Checkpointer on this cylinder, however it was attached."""
    ext = getattr(opt, "extobject", None)
    if ext is None:
        return None
    candidates = list(getattr(ext, "extdict", {}).values()) + [ext]
    for candidate in candidates:
        if type(candidate).__name__ == "Checkpointer":
            return candidate
    return None


def _spoke_marker(wheel):
    """What this spoke restored, or None if it has no Checkpointer.

    A test cannot otherwise tell a restored incumbent from one the spoke
    re-found on its own: farmer is deterministic, so the resumed spoke
    converges on the same answer either way. The same goes for the loop
    cursor, which leaves no trace in the answer at all.
    """
    ext = _checkpointer(wheel.spcomm.opt)
    if ext is None:
        return None
    spoke = wheel.spcomm
    return {
        "cylinder": type(spoke).__name__,
        "strata_rank": spoke.strata_rank,
        # What this spoke holds at the end: the objective of the solution
        # it would write if the hub credits it with the inner bound. None on
        # a dual cylinder, which holds no solution.
        "best_solution_obj_val": getattr(spoke.opt, "best_solution_obj_val",
                                         None),
        "restored_incumbent_obj": ext.restored_incumbent_obj,
        # The iteration a dual cylinder's restored W was written at, or None
        # on a cylinder that has no such state or did not restore any.
        "restored_dual_generation": getattr(ext, "restored_dual_generation",
                                            None),
        # What the loop actually *adopted*, not what the Checkpointer read.
        # The two differ whenever a cursor is read and then refused, and a
        # test that watched the read would score that as a success.
        "applied_loop_state": getattr(spoke, "applied_loop_state", None),
        # Only the xhatter spokes have a loop with a place in it. The dual
        # cylinders run PH, whose iterate is not a cursor.
        "final_loop_state": (spoke.checkpoint_loop_state()
                             if hasattr(spoke, "checkpoint_loop_state")
                             else None),
        "restored_values": _restored_values or None,
    }


def _recording_restore(real_restore):
    """Wrap restore_spoke_incumbent so the cache it builds is recorded."""
    def restore(opt, state):
        obj = real_restore(opt, state)
        for sname, s in opt.local_scenarios.items():
            _restored_values[sname] = {
                var.name: value
                for var, value in s._mpisppy_data.best_solution_cache.items()
            }
        return obj
    return restore


def _recording_is_converged(real_is_converged):
    """Wrap the hub's convergence check so its first inner bound is kept."""
    def is_converged(self, *args, **kwargs):
        _first_inner_bound.setdefault("value", self.BestInnerBound)
        return real_is_converged(self, *args, **kwargs)
    return is_converged


def main():
    if sys.argv[1] != "--out":
        raise RuntimeError("usage: cylinders_ab_driver.py --out PATH [generic_cylinders args]")
    out_path = sys.argv[2]

    captured = {}
    real_do_decomp = generic_cylinders.do_decomp

    def capturing_do_decomp(*args, **kwargs):
        wheel = real_do_decomp(*args, **kwargs)
        captured["wheel"] = wheel
        return wheel

    generic_cylinders.do_decomp = capturing_do_decomp
    ckpt.restore_spoke_incumbent = _recording_restore(
        ckpt.restore_spoke_incumbent)
    PHHub.is_converged = _recording_is_converged(PHHub.is_converged)

    sys.argv = [sys.argv[0]] + sys.argv[3:]
    generic_cylinders.main()

    wheel = captured.get("wheel")
    if wheel is None:
        return
    if wheel.on_hub():
        snapshot = _hub_snapshot(wheel)
        # Rank 0's snapshot also lands at the bare path, which is what the
        # single-rank-per-cylinder harness reads. The per-rank copies are what
        # the multi-rank harness compares, and rank 0 writes both rather than
        # having the two tests disagree about where the hub's answer lives.
        if wheel.cylinder_rank == 0:
            with open(out_path, "w") as f:
                json.dump(snapshot, f)
        with open(f"{out_path}.hubrank{wheel.cylinder_rank:04d}", "w") as f:
            json.dump(snapshot, f)
    else:
        marker = _spoke_marker(wheel)
        if marker is None:
            return
        # Rank 0's marker lands at the ".spoke<strata>" name the
        # single-rank-per-cylinder harnesses read. The per-rank copies go
        # under a different prefix on purpose: those harnesses collect
        # everything named ".spoke*", and a second file per spoke would
        # arrive there as a second spoke.
        if wheel.cylinder_rank == 0:
            with open(f"{out_path}.spoke{wheel.strata_rank}", "w") as f:
                json.dump(marker, f)
        marker["cylinder_rank"] = int(wheel.cylinder_rank)
        with open(f"{out_path}.cyl{wheel.strata_rank}"
                  f"rank{wheel.cylinder_rank:04d}", "w") as f:
            json.dump(marker, f)


if __name__ == "__main__":
    main()
