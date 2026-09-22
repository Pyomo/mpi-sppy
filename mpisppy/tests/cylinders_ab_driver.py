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
        "restored_incumbent_obj": ext.restored_incumbent_obj,
    }


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
