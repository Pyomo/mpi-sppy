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

        Its own method so it can be tested without standing up a full run; the
        guard below it is the kind that only fails on someone else's model.
        """
        problem = None
        # .items(), not .values(): the messages below must name the scenario by
        # its local_scenarios key, as every other guard here does. s.name is
        # the Pyomo model name, which SPBase never sets -- it is whatever the
        # scenario_creator chose, and an unnamed ConcreteModel reports
        # "unknown", which would tell the user nothing.
        for sname, s in self.opt.local_scenarios.items():
            # Existence is not enough: a scenario_creator may already attach an
            # EXPORT or LOCAL `dual` suffix (a common way to supply dual warm
            # starts), and reusing that would import nothing.
            existing = getattr(s, "dual", None)
            if existing is None:
                s.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            elif not isinstance(existing, pyo.Suffix):
                # getattr returns whatever the scenario_creator declared, and
                # `dual` is an ordinary enough name for a Var or Param. Without
                # this the next line raises AttributeError from inside
                # lagrangian_prep, naming neither this spoke nor the component.
                problem = problem or (
                    f"scenario {sname} already has a component named `dual` "
                    f"that is not a Suffix (it is a "
                    f"{type(existing).__name__}); the certificate needs a "
                    "Suffix.IMPORT named `dual` to receive the solver's duals."
                )
            elif not existing.import_enabled():
                problem = problem or (
                    f"scenario {sname} already has a `dual` Suffix that does "
                    "not import; the certificate needs the solver's duals. Use "
                    "Suffix.IMPORT or Suffix.IMPORT_EXPORT."
                )
        self._raise_collectively(problem)

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
        # Every hard error below goes through _raise_collectively, including
        # the two whose conditions are the same on every rank -- self.opt.options
        # comes from the spoke dict and _attach_prox is set identically by
        # PH_Prep. Routing them the same way costs two allreduces once, at
        # setup, and leaves the class one rule instead of a per-guard judgement
        # about whether this particular condition happens to be rank-uniform.
        self._raise_collectively(
            "ipopt_outer_bound requires the proximal term to be off; "
            "the bound it computes is a Lagrangian bound and a proximal "
            "subproblem is not the Lagrangian relaxation"
            if getattr(self.opt, "_attach_prox", False) else None
        )

        solver_name = (self.opt.options.get("solver_name") or "").strip().lower()
        self._raise_collectively(
            f"ipopt_outer_bound is scoped to Ipopt, but its solver is "
            f"{solver_name!r}. The dual sign conventions it relies on have "
            f"been measured only for {sorted(_MEASURED_IPOPT_SOLVERS)}. "
            "Set --ipopt-outer-bound-solver-name."
            if solver_name not in _MEASURED_IPOPT_SOLVERS else None
        )

        # Certifiability is genuinely rank-local -- a discrete variable or a
        # nonlinear equality sits in one scenario on one rank -- so raising
        # here is exactly the case _raise_collectively exists for.
        problem = None
        for sname, s in self.opt.local_scenarios.items():
            try:
                check_model_is_certifiable(s)
            except CertificateError as e:
                problem = problem or f"scenario {sname}: {e}"
        self._raise_collectively(problem)

        # fbbt first (it can only shrink the box, which makes the bound
        # tighter without dropping a feasible point), then say something if a
        # variable is still unbounded. Not an error: this cylinder is an
        # optional source of a bound, and a model that is merely under-bounded
        # is not a broken model. It may simply report nothing.
        still_unbounded = {}
        infeasible = []
        fbbt_failed = []
        for sname, s in self.opt.local_scenarios.items():
            try:
                names = unbounded_variables(s, do_fbbt=True)
            except InfeasibleConstraintException as e:
                # fbbt proved the scenario infeasible. That is the model's
                # problem and the subsequent solve will report it; it is not
                # this spoke's to escalate. Letting it out would MPI_Abort the
                # hub and every other cylinder from a call made to tighten the
                # box and build a diagnostic, which inverts the stand-down
                # policy this spoke states for itself.
                infeasible.append(f"{sname} ({e})")
                continue
            except Exception as e:
                # `except Exception` is deliberate, and naming the classes
                # instead was tried twice and was wrong twice. fbbt raises a
                # whole zoo on models check_model_is_certifiable admits, since
                # a one-sided nonlinear body is the caller's convexity
                # assertion and fbbt has to bound it anyway:
                #   x**y <= 10, x in [-5,-1]   -> IntervalException
                #   x**3 <= y,  x in [1,1e200] -> OverflowError, which is not
                #                                 a PyomoException at all
                # and pyomo.contrib.fbbt.interval raises a bare ValueError on
                # yet another path. Enumerating them tracks Pyomo's interval
                # arithmetic release by release, and every one missed aborts
                # the hub and every other cylinder from a call this spoke makes
                # only to tighten a box and print a diagnostic. The property is
                # that NOTHING raised here is worth ending the run, so catch on
                # that and not on a list.
                #
                # The cost is that a genuine bug in our own code is swallowed
                # too, so the warning reports the exception class and message
                # rather than just a count.
                #
                # Tightening is only ever an improvement, so losing it costs
                # looseness and nothing else. Redo the scan without fbbt -- a
                # pure component_data_objects walk that cannot itself raise --
                # to keep the unbounded-variable diagnostic.
                fbbt_failed.append(f"{sname} ({type(e).__name__}: {e})")
                names = unbounded_variables(s, do_fbbt=False)
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
        self._warn_once_collectively(
            "fbbt_failed",
            bool(fbbt_failed),
            lambda: (
                f"ipopt_outer_bound: bounds tightening could not analyze "
                f"{len(fbbt_failed)} scenario(s) on rank "
                f"{self.cylinder_rank}, for example {fbbt_failed[0]}. Their "
                "variable boxes are used as the model states them, untightened, "
                "so the bound from those scenarios is looser than it could be. "
                "The certificate is unaffected otherwise; this message is "
                "printed once."
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

    def _raise_collectively(self, local_problem):
        """Raise on every rank, or on none.

        The counterpart to _warn_once_collectively, for the conditions that
        are hard errors rather than warnings. Same reason for existing: the
        conditions this spoke checks at setup are rank-local -- a discrete
        variable, a `dual` component of the wrong type -- while everything
        after them is collective. A bare `raise` on the one rank that saw the
        problem leaves the others in the next allreduce with no partner, and
        the run hangs rather than reporting the model error that caused it.
        Today it dies instead of hanging, but only because WheelSpinner.run
        wraps the run in MPI_Abort (#852); driven any other way, or with that
        wrapper bypassed, it is a hang with no traceback.

        `local_problem` is the message for THIS rank's problem, or None. The
        message from the lowest offending rank is broadcast, so every rank's
        traceback names the scenario that actually caused it rather than
        reporting a bare "some other rank failed".

        EVERY rank must call this, including the ranks with nothing to report.
        """
        speaking_rank = self.cylinder_comm.allreduce(
            self.cylinder_rank if local_problem is not None
            else self.cylinder_comm.size,
            op=MPI.MIN,
        )
        if speaking_rank == self.cylinder_comm.size:
            return
        problem = self.cylinder_comm.bcast(local_problem, root=speaking_rank)
        raise CertificateError(f"rank {speaking_rank}: {problem}")

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
            # False so a solution that fails to LOAD is reported rather than
            # raised. spopt hands that case back as solution_available=False,
            # which the loop below already treats as "no bound for this
            # scenario" and now names as a cause -- an optional source of a
            # bound should not take the hub and every other cylinder with it.
            #
            # Worth being exact about the reach of this, because the name
            # oversells it: it governs ONLY the load step. A solve that fails
            # outright still re-raises its solver_exception from the
            # not_good_enough_results branch, which need_solution does not
            # gate. Narrowing that too would take a catch around solve_loop
            # here, which is a bigger change than this flag.
            need_solution=False,
            warmstart=warmstart,
        )

        if self._nonants_newly_fixed():
            for s in self.opt.local_scenarios.values():
                s._mpisppy_data.outer_bound = None
            return self.opt.Ebound(verbose)

        # Keyed by exception CLASS: _warn_once_collectively fires a key once
        # per run, so a single "certificate_failed" key meant the first failure
        # -- typically a routine CertificateError -- consumed the warning and a
        # genuine bug arriving on a later iteration was silent for the rest of
        # the run. That silence is what the `except Exception` below would
        # otherwise buy, and it is not a trade worth making.
        failures_by_class = {}
        no_dual = []
        # Keyed by CAUSE, for the reason failures_by_class is keyed by class:
        # one flat key is consumed by whichever cause happens to arrive first
        # and silences every other one for the rest of the run. The three are
        # distinguishable here and want different advice -- bounding variables
        # fixes one of them and is a wild goose chase for the other two.
        no_bound_by_cause = {}
        # .items() for the same reason as _attach_dual_suffixes: the failure
        # message has to name the scenario by its local_scenarios key. s.name
        # is the Pyomo model name, which SPBase never sets.
        for sname, s in self.opt.local_scenarios.items():
            # solve_loop has just written results.Problem[0].Lower_bound here,
            # which for Ipopt is -inf. Overwrite it with the certificate, or
            # with None when there is no certificate to be had -- Ebound then
            # declines collectively rather than folding a -inf into the sum.
            if not s._mpisppy_data.solution_available:
                # Counted, not just skipped. solve_loop(gripe=True) reports the
                # failed solve, but not the consequence this cylinder is the
                # only one that can state: Ebound is all-or-nothing, so the
                # whole spoke stood down for the iteration, not just this
                # scenario. Leaving it uncounted was the last silent path.
                s._mpisppy_data.outer_bound = None
                no_bound_by_cause.setdefault("no_solution", []).append(sname)
                continue
            # Into a per-scenario list, merged into no_dual only if the call
            # produces A BOUND -- not merely if it returns; see the `else`
            # below. certified_lower_bound extends missing_duals BEFORE the
            # work that decides whether there is a bound at all, so passing
            # no_dual directly let a scenario that produced none report "the
            # bound is looser but still valid" -- false -- and burn the
            # missing_duals warn-once key, hiding a real tightness loss later.
            scenario_no_dual = []
            scenario_reason = []
            try:
                s._mpisppy_data.outer_bound = certified_lower_bound(
                    s, sign_convention="ipopt", eps_rel=self._cushion,
                    missing_duals=scenario_no_dual,
                    no_bound_reason=scenario_reason)
            except Exception as e:
                # `except Exception` for the same reason as the fbbt call in
                # _check_setup_guards, and the enumerated list this replaces
                # was wrong for the same reason. It read:
                #
                #   (CertificateError, ValueError, ArithmeticError)
                #
                # CertificateError being the module's own signal, ValueError
                # what evaluating phi raises on an uninitialized Var or on
                # `math domain error` when bound_relax_factor puts the iterate
                # a hair outside a log or a sqrt, ArithmeticError an overflow.
                # The list looked defensible because dual_certificate is
                # disciplined about its error type -- it converts even the
                # objective finder's RuntimeError into CertificateError. But
                # certified_lower_bound calls Pyomo's differentiate(), which
                # carries no such promise: it raises DifferentiationException,
                # derived straight from Exception, on models this spoke
                # TARGETS. `cosh(x) <= y` is convex and admitted by
                # check_model_is_certifiable, and differentiate has no rule for
                # it; so are sinh, tanh, ceil, floor and Expr_if, and abs(x) at
                # x=0 raises it too. Each one aborted the hub and every other
                # cylinder from inside the iteration loop, every iteration.
                #
                # The property is the one already stated below: no certificate
                # this iteration is not worth ending the run. The exception class is
                # reported so a genuine bug in our own code is still legible.
                failures_by_class.setdefault(
                    type(e).__name__, []).append(f"{sname} ({e})")
                s._mpisppy_data.outer_bound = None
            else:
                # `is not None`, not merely "did not raise". Populating
                # missing_duals happens before the work that decides whether
                # there is a bound at all, and certified_lower_bound RETURNS
                # None on two ordinary outcomes for this spoke -- an unbounded
                # variable with a nonzero gradient component in phi, and a
                # non-finite qhat. Merging on the raise path alone left the
                # same falsehood in place for those: "the bound is looser than
                # it could be but still valid" said of a scenario that has no
                # bound, and the missing_duals key burnt for the run.
                if s._mpisppy_data.outer_bound is not None:
                    no_dual.extend(scenario_no_dual)
                else:
                    # Returning None raises nothing, so without this the
                    # scenario reaches neither failures_by_class nor no_dual
                    # and the run goes completely silent: Ebound declines
                    # collectively and the user reads an empty 'N' column with
                    # nothing said anywhere. Dropping the false "looser but
                    # still valid" message is an improvement only if something
                    # true takes its place.
                    #
                    # Which of the two return-None cases it was comes from
                    # the engine, which knows. Re-deriving it by scanning the
                    # model for an unbounded variable was wrong in one
                    # direction: that scan also finds variables absent from
                    # phi, variables whose gradient component is zero, and
                    # variables unbounded on the side never consulted, so a
                    # non-finite result on a model that merely contains one
                    # was reported as unbounded -- with the advice for the
                    # wrong cause, and the right cause's key left unburnt but
                    # unreachable, since the condition recurs every iteration.
                    # "unclassified", not a guess at one of the known tags:
                    # a return-None site added to certified_lower_bound
                    # without a tag would otherwise print a specific cause,
                    # and advice for it, for something nobody classified.
                    tag, detail = (scenario_reason[0] if scenario_reason
                                   else ("unclassified", "no reason recorded"))
                    no_bound_by_cause.setdefault(tag, []).append(
                        f"{sname} ({detail})")

        # The set of classes to warn about must be GLOBAL. The key drives
        # _warn_once_collectively, and ranks entering it with different keys,
        # or in a different order, is a hang rather than a missed warning --
        # so take the union and walk it sorted.
        seen_here = set(failures_by_class)
        all_classes = sorted(
            set().union(*self.cylinder_comm.allgather(seen_here)))
        for cls in all_classes:
            here = failures_by_class.get(cls, [])
            self._warn_once_collectively(
                f"certificate_failed:{cls}",
                bool(here),
                # Bound as defaults rather than captured. As it stands
                # _warn_once_collectively calls this synchronously, inside
                # this iteration, so a closure would report the right class
                # too; the defaults are what keep that from depending on when
                # the callback runs.
                lambda cls=cls, here=here: (
                    f"ipopt_outer_bound: no certificate ({cls}) for "
                    f"{len(here)} scenario(s) on rank {self.cylinder_rank}, "
                    f"for example {here[0]}. Ebound is all-or-nothing, so this "
                    "cylinder reports NO bound at all on such an iteration, "
                    "not merely for the scenarios named. Some causes are "
                    "structural rather than transient -- an expression "
                    "differentiate has no rule for recurs every iteration -- "
                    "so if the 'N' column stays empty, this is why. Printed "
                    "once per exception class."
                ),
            )
        # The cause set must be global, for the same reason the class set is:
        # the key drives _warn_once_collectively, and ranks entering it with
        # different keys, or in a different order, is a hang.
        # tag -> (what happened, what to do about it). The tags come from
        # certified_lower_bound, except "no_solution", which is this loop's.
        # The advice is per cause because it differs: bounding a variable fixes
        # one of these and is a wild goose chase for the other two.
        _CAUSES = {
            "unbounded_box": (
                "the box minimization was unbounded below",
                "Giving that variable a finite bound is the fix -- the "
                "example above names it and the side it is missing.",
            ),
            "non_finite": (
                "the arithmetic produced a non-finite value",  # detail says which
                "NaN or an infinity in the point or the duals, which usually "
                "means the solve diverged rather than that anything is "
                "unbounded -- bounds will not help.",
            ),
            "unclassified": (
                "the certificate returned no bound and recorded no reason",
                "That is a gap in certified_lower_bound rather than in the "
                "model -- a return-None path that records no tag. Please "
                "report it.",
            ),
            "no_solution": (
                "the solve produced no loadable solution",
                "solve_loop reports the solve itself; what only this cylinder "
                "can say is that the whole spoke stood down, not just that "
                "scenario.",
            ),
        }
        all_tags = sorted(set().union(
            *self.cylinder_comm.allgather(set(no_bound_by_cause))))
        for tag in all_tags:
            here = no_bound_by_cause.get(tag, [])
            # .get, not [tag]: a return-None site recording a NEW tag would
            # otherwise KeyError inside lagrangian() and abort the run every
            # iteration -- worse than the wrong-advice symptom the
            # "unclassified" entry was added to prevent, and the harder half of
            # the same gap. That entry reads correctly for any unknown tag.
            what, advice = _CAUSES.get(tag, _CAUSES["unclassified"])
            self._warn_once_collectively(
                f"no_bound_returned:{tag}",
                bool(here),
                lambda tag=tag, here=here, what=what, advice=advice: (
                    f"ipopt_outer_bound: no bound for {len(here)} scenario(s) "
                    f"on rank {self.cylinder_rank} -- {what}. For example "
                    f"{here[0]}. Ebound is all-or-nothing, so this cylinder "
                    "reports NO bound on such an iteration, not merely for "
                    f"the scenarios named. {advice} Printed once per cause."
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
