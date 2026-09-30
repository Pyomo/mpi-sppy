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


def _hub_snapshot(wheel):
    """The hub's final state, in a shape JSON can hold and a test can diff.

    Keys are strings rather than tuples for the same reason: this crosses a
    process boundary. Values are the iterate itself -- nonant values and
    fixedness, and the per-nonant Params that drive the next iteration -- plus
    the scalars a resume is supposed to carry forward.
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

    def _num(value):
        return None if value is None else float(value)

    return {
        "iteration": int(getattr(opt, "_PHIter", 0)),
        "resumed": bool(getattr(opt, "_resumed_from_checkpoint", False)),
        "resume_iteration": int(getattr(opt, "_resume_iteration", 0)),
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
    converges on the same answer either way.
    """
    ext = _checkpointer(wheel.spcomm.opt)
    if ext is None:
        return None
    return {
        "cylinder": type(wheel.spcomm).__name__,
        "strata_rank": wheel.spcomm.strata_rank,
        # What this spoke holds at the end: the objective of the solution
        # it would write if the hub credits it with the inner bound.
        "best_solution_obj_val": wheel.spcomm.opt.best_solution_obj_val,
        "restored_incumbent_obj": ext.restored_incumbent_obj,
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
        with open(out_path, "w") as f:
            json.dump(_hub_snapshot(wheel), f)
    elif wheel.cylinder_rank == 0:
        marker = _spoke_marker(wheel)
        if marker is not None:
            with open(f"{out_path}.spoke{wheel.strata_rank}", "w") as f:
                json.dump(marker, f)


if __name__ == "__main__":
    main()
