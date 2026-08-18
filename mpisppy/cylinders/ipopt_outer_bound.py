###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# An outer-bound spoke for convex NLP subproblems solved with Ipopt.
#
# The ordinary Lagrangian spoke reads its bound off the solver's dual bound.
# Ipopt is not a branch-and-bound solver and reports none, so that spoke warns
# and produces nothing usable. This one computes the bound itself, from the
# subproblem's own duals, using mpisppy.utils.dual_certificate -- see that
# module for the argument. The short version: the value at the returned point
# is an *inner* bound, and what makes an outer bound available without any
# convergence assumption is Lagrangian weak duality plus a tangent-plane
# underestimator minimized in closed form over the variable box.
#
# Convexity of the scenario subproblems is the user's assertion. The parts of it
# that can be checked mechanically are checked at setup and are hard errors; see
# _check_setup_guards.

import warnings

import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
from mpisppy.cylinders.lagrangian_bounder import _LagrangianMixin
from mpisppy.cylinders.spoke import OuterBoundWSpoke
from mpisppy.utils.dual_certificate import (
    CertificateError,
    certified_lower_bound,
    check_model_is_certifiable,
    unbounded_variables,
)


class IpoptOuterBound(_LagrangianMixin, OuterBoundWSpoke):

    converger_spoke_char = 'I'

    # The certificate reads the point *and* the duals off the solved model, so
    # unlike the Lagrangian spoke this one cannot skip loading the solution.
    outer_bound_only = False

    def ipopt_outer_bound_prep(self):
        """lagrangian_prep plus what the certificate needs: a dual suffix on
        every subproblem, the setup guards, and the bound tightening."""
        self.lagrangian_prep()

        # PH_Prep(attach_prox=False) is what makes this the Lagrangian
        # relaxation rather than a proximal subproblem. Assert rather than
        # trust: with a prox term the number below is not a Lagrangian bound.
        if not getattr(self.opt, "prox_disabled", True) and not self.opt.W_disabled:
            raise CertificateError(
                "ipopt_outer_bound requires the proximal term to be absent; "
                "the bound it computes is a Lagrangian bound and a proximal "
                "subproblem is not the Lagrangian relaxation"
            )

        for s in self.opt.local_scenarios.values():
            if not hasattr(s, "dual"):
                s.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        self._check_setup_guards()

        # Snapshot which nonants are fixed now, so a fixing extension that
        # fixes more of them later can be caught: fixing a nonant restricts the
        # subproblem, which can only raise its minimum, so the result would
        # bound the restricted problem and not the original. Nonants the
        # scenario creator fixed are part of the problem and are fine.
        self._fixed_at_setup = {
            (sname, ndn_i): xvar.fixed
            for sname, s in self.opt.local_scenarios.items()
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items()
        }

    def _check_setup_guards(self):
        """Hard errors for the parts of the theorem that are checkable, and a
        warning for the part that only costs tightness."""
        solver_name = self.opt.options.get("solver_name") or ""
        if "ipopt" not in solver_name:
            raise CertificateError(
                f"ipopt_outer_bound is scoped to Ipopt, but its solver is "
                f"{solver_name!r}. The dual sign conventions it relies on are "
                "measured from Ipopt only. Set --ipopt-outer-bound-solver-name."
            )

        for sname, s in self.opt.local_scenarios.items():
            try:
                check_model_is_certifiable(s)
            except CertificateError as e:
                raise CertificateError(f"scenario {sname}: {e}") from None

        # fbbt first (it can only shrink the box, which makes the bound
        # tighter without dropping a feasible point), then say something if a
        # variable is still unbounded. Not an error: this cylinder is an
        # optional source of a bound, and a model that is merely under-bounded
        # is not a broken model. It may simply report nothing.
        still_unbounded = {}
        for sname, s in self.opt.local_scenarios.items():
            names = unbounded_variables(s, do_fbbt=True)
            if names:
                still_unbounded[sname] = names
        if still_unbounded and self.cylinder_rank == 0:
            sname, names = next(iter(still_unbounded.items()))
            warnings.warn(
                f"ipopt_outer_bound: {len(still_unbounded)} scenario(s) have "
                "variables with no finite bound after fbbt, for example "
                f"{sname}: {', '.join(names[:5])}"
                f"{' ...' if len(names) > 5 else ''}. The certificate minimizes "
                "over the variable box, so an unbounded direction with a "
                "nonzero gradient yields no bound and this spoke will stay "
                "quiet on those iterations. Bounding those variables is what "
                "makes this spoke useful."
            )

    def _assert_nonants_not_newly_fixed(self):
        newly = [
            f"{sname}:{ndn_i}"
            for sname, s in self.opt.local_scenarios.items()
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items()
            if xvar.fixed and not self._fixed_at_setup[(sname, ndn_i)]
        ]
        if newly:
            raise CertificateError(
                "ipopt_outer_bound: nonanticipative variables were fixed after "
                f"setup ({', '.join(newly[:5])}"
                f"{' ...' if len(newly) > 5 else ''}). Fixing restricts the "
                "subproblem, so its minimum is a bound on the restricted "
                "problem and not on the original. Remove the fixing extension "
                "from this spoke."
            )

    def _solve_and_certify(self, warmstart=sputils.WarmstartStatus.PRIOR_SOLUTION):
        """Solve every subproblem, then replace the solver's (useless) bound
        with the certificate. Returns the expected outer bound, or None."""
        self.receive_nonant_bounds()
        verbose = self.opt.options['verbose']
        teeme = self.opt.options.get('tee-rank0-solves', False)

        self.opt.solve_loop(
            solver_options=self.opt._effective_solver_options(self.opt._PHIter),
            dtiming=False,
            gripe=True,
            tee=teeme,
            verbose=verbose,
            need_solution=True,
            warmstart=warmstart,
        )

        self._assert_nonants_not_newly_fixed()

        for s in self.opt.local_scenarios.values():
            # solve_loop has just written results.Problem[0].Lower_bound here,
            # which for Ipopt is -inf. Overwrite it with the certificate, or
            # with None when there is no certificate to be had -- Ebound then
            # declines collectively rather than folding a -inf into the sum.
            if not s._mpisppy_data.solution_available:
                s._mpisppy_data.outer_bound = None
                continue
            s._mpisppy_data.outer_bound = certified_lower_bound(
                s, sign_convention="ipopt", eps_rel=self._cushion)

        return self.opt.Ebound(verbose)

    @property
    def _cushion(self):
        return self.opt.options.get("ipopt_outer_bound_cushion", 1e-9)

    def _set_weights_and_solve(self, warmstart=sputils.WarmstartStatus.PRIOR_SOLUTION):
        self.opt.W_from_flat_list(self.localWs)
        return self._solve_and_certify(warmstart=warmstart)

    def main(self):
        self.verbose = self.opt.options['verbose']
        extensions = self.opt.extensions is not None

        self.ipopt_outer_bound_prep()

        if extensions:
            self.opt.extobject.pre_iter0()

        self.opt._PHIter = 0
        self.trivial_bound = self._solve_and_certify(
            warmstart=sputils.WarmstartStatus.USER_SOLUTION)

        if extensions:
            self.opt.extobject.post_iter0()
        self.opt._PHIter += 1
        self.opt.current_solver_options = {}

        if self.trivial_bound is not None:
            self.send_bound(self.trivial_bound)
        if extensions:
            self.opt.extobject.post_iter0_after_sync()

        while not self.got_kill_signal():
            if self.update_Ws():
                if extensions:
                    self.opt.extobject.miditer()
                bound = self._set_weights_and_solve()
                if extensions:
                    self.opt.extobject.enditer()
                if bound is not None:
                    self.send_bound(bound)
                if extensions:
                    self.opt.extobject.enditer_after_sync()
                self.opt._PHIter += 1
