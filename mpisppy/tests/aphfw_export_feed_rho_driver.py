###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Which x the rho extensions see, and why it matters (design/fw-export-split.md).

`opt.py:_sync_cpp_to_pyomo` feeds mpi-sppy's extensions `get_all_local_prox_x`,
the proximal minimizer the algorithm iterates on, rather than
`get_all_local_nonants`, which exports a vertex of K_s for the xhat spokes.

NormRhoUpdater is where the difference shows.  Its primal residual is
sum_s p_s |x_s - xbar|, and its third branch -- "Converged, Decreasing" --
needs both that and the dual residual under the convergence tolerance.  The
proximal minimizer lives in conv K_s, so on an integer problem it can approach
the fractional consensus point and the residual can reach zero.  An exported
vertex is an extreme point of K_s and cannot, so under the old feed the
residual is floored away from zero however well the run converges and that
branch is unreachable.

This runs the integer farmer both ways and checks exactly that.

Must be run under MPI with exactly 3 ranks:
    mpirun -np 3 python -m mpi4py aphfw_export_feed_rho_driver.py
"""

import sys

from mpi4py import MPI

from mpisppy.extensions.extension import MultiExtension
from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
from mpisppy.opt.aphfw import APHFWOpt
from mpisppy.tests.examples.farmer import (
    scenario_creator,
    scenario_denouement,
)
from mpisppy.tests.utils import aphfw_test_solver

NUM_SCENS = 3
MAX_CYCLES = 40

# NormRhoUpdater's own convergence_tolerance, loosened from its 1e-4 default.
# Forty cycles of the integer farmer do not bring either feed's residuals to
# 1e-4, so at the default neither reaches the converged branch and the
# comparison says nothing.  What the two feeds do reach is far apart: the
# prox-x feed takes the branch at any tolerance above about 2.6e-2, the export
# feed not until 1.1e+1.  0.1 sits in that window with room on both sides.  The
# tolerance feeds back into the rho updates and so moves the thresholds a
# little, which is why the test asserts the bracket rather than the number.
CONVERGENCE_TOL = 0.1

# The export feed's primal residual must stay at least this many times the
# prox-x feed's.  The observed ratio is around 900; 100 is the floor that
# still means "orders of magnitude apart" rather than "slightly worse".  This
# is the mechanism stated without reference to any tolerance.
SEPARATION = 100.0


class RecordingNormRhoUpdater(NormRhoUpdater):
    """NormRhoUpdater that also keeps the residuals each miditer acted on.

    The residuals are recomputed with the extension's own helpers immediately
    before `super().miditer()` runs, and `_prev_avg` -- the only state the dual
    residual depends on -- is not touched until `_snapshot_avg` inside that
    call.  So these are the same numbers the branch tests see, not a
    re-derivation of them.
    """

    instances = []

    def __init__(self, ph):
        super().__init__(ph)
        self.records = []
        RecordingNormRhoUpdater.instances.append(self)

    def miditer(self):
        had_prev = bool(self._prev_avg)
        if had_prev:
            primal = self._compute_primal_residual_norm(self.ph)
            dual = self._compute_dual_residual_norm(self.ph)
        super().miditer()
        if had_prev:
            self.records.append((primal, dual))

    def branch_counts(self):
        """How many (miditer, nonant) pairs took each of the three branches."""
        tol = self._tol
        factor = self._primal_dual_difference_factor
        counts = {"increase": 0, "decrease": 0, "converged": 0}
        for primal, dual in self.records:
            for k, p in primal.items():
                d = dual[k]
                if p > factor * d and p > tol:
                    counts["increase"] += 1
                elif d > factor * p and d > tol:
                    counts["decrease"] += 1
                elif p < tol and d < tol:
                    counts["converged"] += 1
        return counts

    def smallest_primal_residual(self):
        return min((min(p.values()) for p, _ in self.records), default=float("nan"))

    def tolerance_that_would_converge(self):
        """The smallest tolerance at which some nonant takes the third branch.

        Reaching it needs primal < tol and dual < tol on the same nonant in the
        same miditer, and when that holds the two earlier branches cannot fire
        (each requires its own residual to exceed tol).  So the branch fires
        exactly when max(primal, dual) < tol, and the smallest such max over
        every (miditer, nonant) pair is the threshold.
        """
        return min((max(p[k], d[k]) for p, d in self.records for k in p),
                   default=float("nan"))

    def smallest_dual_residual(self):
        return min((min(d.values()) for _, d in self.records), default=float("nan"))


class _ExportFeedShim:
    """Delegates to the real binding, but answers the prox-x accessor with the
    exported vertices -- i.e. what the extensions saw before the export split.
    """

    def __init__(self, cpp):
        object.__setattr__(self, "_cpp", cpp)

    def get_all_local_prox_x(self, buf):
        self._cpp.get_all_local_nonants(buf)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_cpp"), name)


class ExportFeedOpt(APHFWOpt):
    """APHFWOpt with the shipped build's feed: extensions read the export."""

    def _sync_cpp_to_pyomo(self):
        if self._cpp is not None and not isinstance(self._cpp, _ExportFeedShim):
            self._cpp = _ExportFeedShim(self._cpp)
        super()._sync_cpp_to_pyomo()


def _make_options():
    all_scenario_names = [f"Scenario{i}" for i in range(NUM_SCENS)]
    options = {
        "solver_name": aphfw_test_solver(),
        "defaultPHrho": 1.0,
        "convthresh": 0,
        "PHIterLimit": 0,
        "time_limit": 600,
        "verbose": False,
        "display_progress": False,
        "display_convergence_detail": False,
        "iter0_solver_options": {},
        "iterk_solver_options": {},
        "tee-rank0-solves": False,
        "trace_prefix": "",
        "presolve": False,
        "rounding_bias": 0.0,
        "warmstart_subproblems": False,
        "user_warmstart": False,
        "norm_rho_options": {"verbose": False,
                             "convergence_tolerance": CONVERGENCE_TOL},
        # aphfw-specific
        "aphfw_fw_tmax": 2,
        "aphfw_fw_tau": 0.01,
        "aphfw_max_cycles": MAX_CYCLES,
        # Loose enough that APH does not stop before the residuals get small.
        "aphfw_primal_term_thresh": 1e-8,
        "aphfw_dual_term_thresh": 1e-8,
        "aphfw_display_progress": False,
        "aphfw_block_sync": True,
        "aphfw_verbosity": 0,
    }
    kwargs = dict(
        all_scenario_names=all_scenario_names,
        scenario_creator=scenario_creator,
        scenario_denouement=scenario_denouement,
        # The integer farmer: the design note's instance for this check.
        scenario_creator_kwargs={"num_scens": NUM_SCENS, "use_integer": True},
        extensions=MultiExtension,
        extension_kwargs={"ext_classes": [RecordingNormRhoUpdater]},
    )
    return options, kwargs


def _run(opt_class):
    RecordingNormRhoUpdater.instances = []
    options, kwargs = _make_options()
    opt = opt_class(options, **kwargs)
    opt.aphfw_main(finalize=False)
    cycles = opt._cpp.get_current_cycle()
    ext = RecordingNormRhoUpdater.instances[-1]
    counts = ext.branch_counts()
    smallest = ext.smallest_primal_residual()
    smallest_dual = ext.smallest_dual_residual()
    threshold = ext.tolerance_that_would_converge()
    opt.aphfw_finalize()
    return cycles, counts, smallest, smallest_dual, threshold, ext._tol


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() != NUM_SCENS:
        if rank == 0:
            print(f"ERROR: this test requires exactly {NUM_SCENS} MPI ranks, "
                  f"got {comm.Get_size()}")
        return 1

    prox_cycles, prox_counts, prox_min, prox_dual, prox_thr, tol = _run(APHFWOpt)
    exp_cycles, exp_counts, exp_min, exp_dual, exp_thr, _ = _run(ExportFeedOpt)

    if rank == 0:
        print("\n=== NormRhoUpdater branches, integer farmer, "
              f"tol={tol} ===")
        print(f"  prox-x feed   cycles={prox_cycles} {prox_counts} "
              f"min primal={prox_min:.3e} min dual={prox_dual:.3e} "
              f"converges at tol>{prox_thr:.3e}")
        print(f"  export feed   cycles={exp_cycles} {exp_counts} "
              f"min primal={exp_min:.3e} min dual={exp_dual:.3e} "
              f"converges at tol>{exp_thr:.3e}")
        # prox_min can legitimately reach zero -- that is the hypothesis, the
        # proximal minimizer lies in conv K_s and can meet the consensus point
        # -- so the ratio is reported as unbounded rather than dividing by it.
        # The assertion below is written as a product for the same reason.
        if prox_min > 0.0:
            ratio = f"{exp_min / prox_min:.1f}"
        else:
            ratio = "inf (the prox-x feed's residual reached zero)"
        print(f"  primal residual ratio export/prox-x = {ratio} "
              f"(must exceed {SEPARATION:g})")

    ok = True
    if prox_counts["converged"] <= 0:
        ok = False
        if rank == 0:
            print("  FAIL: the both-residuals-converged branch was never "
                  "reached under the prox-x feed.")
    if exp_counts["converged"] != 0:
        ok = False
        if rank == 0:
            print("  FAIL: the both-residuals-converged branch was reached "
                  "under the export feed, so the two feeds are not "
                  "distinguishable on this instance.")
    # The chosen tolerance has to sit strictly between the two feeds' thresholds,
    # or the result above is an accident of where it was put.
    if not (prox_thr < CONVERGENCE_TOL < exp_thr):
        ok = False
        if rank == 0:
            print(f"  FAIL: convergence_tolerance {CONVERGENCE_TOL:g} does not "
                  f"lie between the prox-x threshold {prox_thr:.3e} and the "
                  f"export threshold {exp_thr:.3e}.")
    # The mechanism itself, independent of where the tolerance was put.
    if not (exp_min > SEPARATION * prox_min):
        ok = False
        if rank == 0:
            print(f"  FAIL: the export feed's primal residual, {exp_min:.3e}, is "
                  f"not {SEPARATION:g}x the prox-x feed's, {prox_min:.3e}; the "
                  "two feeds are not meaningfully separated on this instance.")

    if rank == 0:
        print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
