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

from pyomo.contrib.fbbt.fbbt import InfeasibleConstraintException

from mpisppy import MPI

import mpisppy.utils.sputils as sputils
from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
from mpisppy.utils.dual_certificate import (
    CertificateError,
    certified_lower_bound,
    check_model_is_certifiable,
    unbounded_variables,
)


# Solver names whose dual sign conventions have actually been measured against
# this certificate. A substring test was tried first and was too permissive: it
# admitted ipopt_v2 and appsi_ipopt, whose writer runs a linear presolve that
# eliminates rows and then cannot load their duals, and cyipopt, whose
# convention has never been checked. Each of those fails at solve time with an
# error that names something other than the solver choice that caused it.
_MEASURED_IPOPT_SOLVERS = frozenset({"ipopt"})


class IpoptOuterBound(LagrangianOuterBound):
    """The Lagrangian outer-bound spoke with the bound computed, not read.

    Subclasses rather than forks: everything about driving the cylinder --
    iter0, the W loop, extensions, the wait branch, _PreLoopXhatMixin -- is
    identical, and the single difference is where the number comes from. That
    difference lives in lagrangian(), which is the one method overridden.
    """


    # 'N' for NLP. Not 'I': that is InnerBoundSpoke's character, and the hub
    # prints the outer and inner chars side by side, so an 'I' in the outer
    # column would read as an inner bound.
    converger_spoke_char = 'N'

    # The certificate reads the point *and* the duals off the solved model, so
    # unlike the Lagrangian spoke this one cannot skip loading the solution.
    outer_bound_only = False

    def lagrangian_prep(self):
        """The base prep plus what the certificate needs: a dual suffix on
        every subproblem, the setup guards, and the bound tightening.

        Overriding the hook rather than adding a second one is what lets main()
        be inherited unchanged."""
        super().lagrangian_prep()

        self._attach_dual_suffixes()

        self._warned = set()
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

    def _attach_dual_suffixes(self):
        """Give every subproblem a dual Suffix that actually imports.

        Its own method so it can be tested without standing up a wheel; the
        guard below it is the kind that only fails on someone else's model.
        """
        for s in self.opt.local_scenarios.values():
            # Existence is not enough: a scenario_creator may already attach an
            # EXPORT or LOCAL `dual` suffix (a common way to supply dual warm
            # starts), and reusing that would import nothing.
            existing = getattr(s, "dual", None)
            if existing is None:
                s.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            elif not existing.import_enabled():
                raise CertificateError(
                    f"scenario {s.name} already has a `dual` Suffix that does "
                    "not import; the certificate needs the solver's duals. Use "
                    "Suffix.IMPORT or Suffix.IMPORT_EXPORT."
                )

    def _check_setup_guards(self):
        """Hard errors for the parts of the theorem that are checkable, and a
        warning for the part that only costs tightness."""
        # A TRIPWIRE, not a runtime check, and it is worth being honest about
        # which. The bound is a Lagrangian bound, so a proximal subproblem
        # would not produce it. _attach_prox is the right predicate -- it
        # records whether a prox term was actually spliced into the objective
        # -- but the only path here is _LagrangianMixin.lagrangian_prep, which
        # hardcodes PH_Prep(attach_prox=False), so under the base class as it
        # stands this cannot fire. It exists to fail loudly if that hardcoded
        # argument ever changes, which is a real risk on inherited code.
        #
        # It replaced a test on prox_on that was wrong rather than merely
        # unreachable: that Param is created at 0 by attach_Ws_and_prox so it
        # could never fire either, AND _reenable_prox sets it to 1 whether or
        # not the objective contains a prox term, so it would also have fired
        # on models that have none.
        if getattr(self.opt, "_attach_prox", False):
            raise CertificateError(
                "ipopt_outer_bound requires the proximal term to be off; "
                "the bound it computes is a Lagrangian bound and a proximal "
                "subproblem is not the Lagrangian relaxation"
            )

        solver_name = (self.opt.options.get("solver_name") or "").strip().lower()
        if solver_name not in _MEASURED_IPOPT_SOLVERS:
            raise CertificateError(
                f"ipopt_outer_bound is scoped to Ipopt, but its solver is "
                f"{solver_name!r}. The dual sign conventions it relies on have "
                f"been measured only for {sorted(_MEASURED_IPOPT_SOLVERS)}. "
                "Set --ipopt-outer-bound-solver-name."
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
        infeasible = []
        for sname, s in self.opt.local_scenarios.items():
            try:
                names = unbounded_variables(s, do_fbbt=True)
            except InfeasibleConstraintException as e:
                # fbbt proved the scenario infeasible. That is the model's
                # problem and the subsequent solve will report it; it is not
                # this spoke's to escalate. Letting it out would MPI_Abort the
                # hub and every other cylinder from a call made to tighten the
                # box and build a diagnostic, which is exactly the stand-down
                # stand-down policy this spoke states for itself, inverted.
                infeasible.append(f"{sname} ({e})")
                continue
            if names:
                still_unbounded[sname] = names
        self._warn_once_collectively(
            "fbbt_infeasible",
            bool(infeasible),
            lambda: (
                f"ipopt_outer_bound: bounds tightening found {len(infeasible)} "
                f"scenario(s) infeasible on rank {self.cylinder_rank}, for "
                f"example {infeasible[0]}. An infeasible scenario never yields "
                "a certificate and Ebound is all-or-nothing, so this spoke "
                "will report NO bound for the entire run, not just for those "
                "scenarios. The solve will report the infeasibility itself; "
                "this message is printed once."
            ),
        )
        def _unbounded_message():
            sname, names = next(iter(still_unbounded.items()))
            return (
                f"ipopt_outer_bound: {len(still_unbounded)} scenario(s) have "
                "variables with no finite bound after fbbt, for example "
                f"{sname}: {', '.join(names[:5])}"
                f"{' ...' if len(names) > 5 else ''}. This is a heads-up, not "
                "a prediction: the certificate minimizes over the variable "
                "box, so such a variable costs a bound only on an iteration "
                "where its gradient component in phi points along the "
                "unbounded direction. At a KKT point that component is zero, "
                "so a converging solve typically reports a bound anyway. If "
                "the 'N' column does come up empty, bounding these is the fix."
            )

        # Collective: a scenario with an unbounded variable sits on one rank,
        # and warning only when that rank happens to be rank 0 loses the
        # message on every other layout.
        self._warn_once_collectively(
            "unbounded_variables", bool(still_unbounded), _unbounded_message)

    def _warn_once_collectively(self, key, local_flag, message_from_rank):
        """Warn once, from the lowest rank that saw the condition.

        This spoke is an optional source of a bound, so it complains and stands
        down rather than taking the run with it -- an exception from inside the
        iteration loop would MPI_Abort the hub and every other spoke.

        `message_from_rank` is called only on the rank that ends up speaking,
        so building the message stays cheap on the ranks that do not. Note
        that it therefore reports THAT RANK's counts; the reduction settles
        who speaks, not what the totals are, so messages say which rank they
        describe rather than implying a global figure. The
        allreduce is what keeps the diagnostic honest: every condition in this
        class is rank-local while Ebound is collective, so a plain
        `if cylinder_rank == 0` gate would silence the one rank that saw the
        problem and leave the user an empty bound column with no explanation.

        EVERY rank must call this, including the ranks with nothing to report
        -- the allreduce is what makes it work, and a caller that skips it
        behind a rank-local `if` hangs the run instead of warning.

        Returns True if ANY rank saw the condition. That answer is global, so
        it is safe to branch on; branching on the rank-local flag instead is
        what puts different ranks into different collectives.
        """
        # The reduction is unconditional -- not behind the _warned check --
        # because the return value is a GLOBAL answer that callers branch on.
        # Skipping it on the second call would both diverge the ranks and hand
        # back a rank-local answer.
        speaking_rank = self.cylinder_comm.allreduce(
            self.cylinder_rank if local_flag else self.cylinder_comm.size,
            op=MPI.MIN,
        )
        anyone = speaking_rank < self.cylinder_comm.size
        if anyone and key not in self._warned:
            # `anyone` is global, so every rank adds the key together and
            # _warned stays identical across ranks.
            self._warned.add(key)
            if self.cylinder_rank == speaking_rank:
                warnings.warn(message_from_rank())
        return anyone

    def _nonants_newly_fixed(self):
        """True if ANY rank fixed a nonant since setup, in which case no bound
        can be reported: fixing restricts the subproblem, so its minimum bounds
        the restricted problem and not the original.

        The answer is global on purpose. Callers branch on it, and a
        rank-local answer would send some ranks down a path that skips
        collectives the others enter -- a hang. Ebound is all-or-nothing
        anyway, so one rank's fixed nonant already silences the cylinder.

        Deliberately tests `.fixed` and NOT the variable bounds, though
        receive_nonant_bounds can narrow a nonant's box all the way to a single
        point without ever touching `.fixed`. That looks like the same event
        and is not. Narrowing through that channel carries the weak-form
        argument set out in _solve_and_certify -- reduced_costs_spoke's
        contract that an optimal solution survives, applied identically to
        every scenario -- whereas an extension calling fix() carries no
        argument at all. Extending this check to the bounds would make the
        spoke silent whenever --reduced-costs is running, which is the
        combination the weak-form argument exists to permit.
        """
        newly = [
            f"{sname}:{ndn_i}"
            for sname, s in self.opt.local_scenarios.items()
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items()
            if xvar.fixed and not self._fixed_at_setup[(sname, ndn_i)]
        ]
        return self._warn_once_collectively(
            "fixed_nonants",
            bool(newly),
            lambda:
            "ipopt_outer_bound: nonanticipative variables were fixed after "
            f"setup ({', '.join(newly[:5])}"
            f"{' ...' if len(newly) > 5 else ''}). Fixing restricts the "
            "subproblem, so its minimum is a bound on the restricted problem "
            "and not on the original. This spoke will report no bound until "
            "they are unfixed; remove the fixing extension from this spoke."
        )

    def lagrangian(self, warmstart=sputils.WarmstartStatus.PRIOR_SOLUTION):
        """Solve every subproblem, then replace the solver's (useless) bound
        with the certificate. Returns the expected outer bound, or None.

        This is the whole of the difference from the base spoke, which reads
        results.Problem[0].Lower_bound instead -- the number Ipopt does not
        provide."""
        # This shrinks the box the certificate minimizes over, so it needs an
        # argument. Note the tempting one -- "a smaller box removes points, and
        # fewer points can only raise an infimum" -- is an argument that the
        # bound gets TIGHTER, which is precisely the direction that could break
        # it. What is actually needed is that every scenario's box still holds
        # one COMMON optimal solution x* of the full problem: then the
        # certificate is below phi_s(x*) <= f_s(x*) + W_s'x* for each s, and the
        # p-weighted sum is below OPT. The fbbt done at setup gives this the
        # easy way, by removing no feasible point at all. This channel does not
        # -- only reduced_costs_spoke sends it, and reduced-cost fixing does
        # discard feasible points -- so it rests on that spoke's own contract
        # that an optimal solution survives, plus the fact that the bounds are
        # broadcast and applied identically to every scenario, which is what
        # makes the surviving solution common rather than per-scenario. A
        # sender that guarantees neither would break the bound silently.
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

        if self._nonants_newly_fixed():
            for s in self.opt.local_scenarios.values():
                s._mpisppy_data.outer_bound = None
            return self.opt.Ebound(verbose)

        failures = []
        no_dual = []
        for s in self.opt.local_scenarios.values():
            # solve_loop has just written results.Problem[0].Lower_bound here,
            # which for Ipopt is -inf. Overwrite it with the certificate, or
            # with None when there is no certificate to be had -- Ebound then
            # declines collectively rather than folding a -inf into the sum.
            if not s._mpisppy_data.solution_available:
                s._mpisppy_data.outer_bound = None
                continue
            try:
                s._mpisppy_data.outer_bound = certified_lower_bound(
                    s, sign_convention="ipopt", eps_rel=self._cushion,
                    missing_duals=no_dual)
            except (CertificateError, ValueError, ArithmeticError) as e:
                # CertificateError is the module's own signal (e.g. a routine
                # solver outcome leaving a constraint without a dual). The
                # other two are what evaluating phi and its gradient at the
                # returned point can raise on the very class of model this
                # spoke targets: ValueError from an uninitialized Var or from
                # `math domain error` when bound_relax_factor puts the iterate
                # a hair outside a bound under a log or a sqrt, ArithmeticError
                # from an overflow. All three mean the same thing here -- no
                # certificate this iteration -- and none is worth taking down
                # the hub and every other spoke from inside the iteration loop.
                failures.append(f"{s.name} ({e})")
                s._mpisppy_data.outer_bound = None

        self._warn_once_collectively(
            "certificate_failed",
            bool(failures),
            lambda: (
                f"ipopt_outer_bound: no certificate for {len(failures)} "
                f"scenario(s) on rank {self.cylinder_rank}, for example "
                f"{failures[0]}. Ebound is all-or-nothing, so this cylinder "
                "reports no bound at all this iteration -- not merely for the "
                "scenarios named."
            ),
        )
        self._warn_once_collectively(
            "missing_duals",
            bool(no_dual),
            lambda: (
                f"ipopt_outer_bound: {len(no_dual)} constraint(s) had no dual "
                f"imported, for example {no_dual[0]}. They are taken with "
                "multiplier zero, which weak duality admits, so the bound is "
                "looser than it could be but still valid."
            ),
        )
        return self.opt.Ebound(verbose)

    @property
    def _cushion(self):
        return self.opt.options.get("ipopt_outer_bound_cushion", 1e-9)

    def _jensens_enabled(self):
        """Never, for this spoke.

        The inherited main() offers a Jensen's bound before the loop, and
        _jensens_solve takes it from results.problem.lower_bound -- the
        solver's own dual bound. That is precisely the number Ipopt does not
        produce, and the reason this spoke exists. Taking it would send a
        meaningless bound, so the step is declined rather than inherited.
        """
        return False
