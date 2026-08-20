###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Test PH.ph_main() directly (not through the cylinder hub system).
# This exercises the standalone PH code path, including extensions
# and convergers that are otherwise untested.

import os
import shutil
import unittest

import pyomo.environ as pyo
import mpisppy.opt.ph
import mpisppy.tests.examples.farmer as farmer
from mpisppy.tests.examples.sizes.sizes import scenario_creator as sizes_creator, \
                                               scenario_denouement as sizes_denouement, \
                                               id_fix_list_fct
from mpisppy.tests.utils import get_solver
import mpisppy.MPI as mpi

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

# Known reference values (EF optimal for farmer with 3 scenarios, crops_multiplier=1)
FARMER_EF_OBJ = -118361.33  # approximate


class TestPHMainFarmer(unittest.TestCase):
    """Test PH.ph_main() with the farmer model, checking solution quality."""

    def setUp(self):
        self.options = {
            "solver_name": solver_name,
            "PHIterLimit": 50,
            "defaultPHrho": 1,
            "convthresh": 1e-8,
            "verbose": False,
            "display_timing": False,
            "display_progress": False,
            "iter0_solver_options": {"threads": 1},
            "iterk_solver_options": {"threads": 1},
            "smoothed": 0,
            "asynchronousPH": False,
            "toc": False,
        }
        self.scenario_names = [f"Scenario{i+1}" for i in range(3)]
        self.creator_kwargs = {"crops_multiplier": 1}

    def _copy_options(self):
        return dict(self.options)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_obj(self):
        """PH on farmer should approach the EF optimal."""
        ph = mpisppy.opt.ph.PH(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj, tbound = ph.ph_main()
        # obj includes the proximal term so won't match EF exactly,
        # but should be in the right ballpark
        self.assertAlmostEqual(obj, FARMER_EF_OBJ, delta=500)
        # trivial bound should be looser (more negative for min)
        self.assertLess(tbound, FARMER_EF_OBJ)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_nonants_converge(self):
        """After PH, farmer nonants should be close across scenarios."""
        ph = mpisppy.opt.ph.PH(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj, tbound = ph.ph_main()
        # Collect first-stage decisions from all local scenarios
        nonant_values = {}
        for sname, s in ph.local_scenarios.items():
            for node in s._mpisppy_node_list:
                for i, v in enumerate(node.nonant_vardata_list):
                    key = (node.name, i)
                    if key not in nonant_values:
                        nonant_values[key] = []
                    nonant_values[key].append(pyo.value(v))
        # Check nonants are close across scenarios (convergence)
        for key, vals in nonant_values.items():
            spread = max(vals) - min(vals)
            self.assertLess(spread, 1.0,
                            f"Nonant {key} spread {spread} too large")

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_no_finalize(self):
        """PH.ph_main() with finalize=False returns None for Eobj."""
        ph = mpisppy.opt.ph.PH(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj, tbound = ph.ph_main(finalize=False)
        self.assertIsNone(obj)
        self.assertIsNotNone(tbound)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_xhatlooper_obj(self):
        """PH with XhatLooper should find a good xhat."""
        from mpisppy.extensions.xhatlooper import XhatLooper
        options = self._copy_options()
        options["xhat_looper_options"] = {
            "xhat_solver_options": None,
            "scen_limit": 3,
        }
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            extensions=XhatLooper,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertAlmostEqual(obj, FARMER_EF_OBJ, delta=500)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_frac_converger(self):
        """FractionalConverger on farmer (LP) converges immediately."""
        from mpisppy.convergers.fracintsnotconv import FractionalConverger
        ph = mpisppy.opt.ph.PH(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            ph_converger=FractionalConverger,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)
        # With no integers, converger says "converged" after iter 0,
        # so PH stops at iteration 1 — obj won't be close to optimal
        # but the trivial bound should still be valid
        self.assertLess(tbound, FARMER_EF_OBJ)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_diagnoser(self):
        """Diagnoser should create per-scenario output files."""
        from mpisppy.extensions.diagnoser import Diagnoser
        diagdir = os.path.join(os.path.dirname(__file__), "_test_diagdir")
        if os.path.exists(diagdir):
            shutil.rmtree(diagdir)
        options = self._copy_options()
        options["PHIterLimit"] = 3
        options["diagnoser_options"] = {"diagnoser_outdir": diagdir}
        try:
            ph = mpisppy.opt.ph.PH(
                options,
                self.scenario_names,
                farmer.scenario_creator,
                farmer.scenario_denouement,
                scenario_creator_kwargs=self.creator_kwargs,
                extensions=Diagnoser,
            )
            conv, obj, tbound = ph.ph_main()
            self.assertIsNotNone(obj)
            # Check that diagnostic files were created
            for sname in self.scenario_names:
                dag_file = os.path.join(diagdir, f"{sname}.dag")
                self.assertTrue(os.path.exists(dag_file),
                                f"Missing diagnostic file for {sname}")
                with open(dag_file) as f:
                    lines = f.readlines()
                # header + iter0 write + 3 enditer writes = 4+ lines
                self.assertGreater(len(lines), 1)
        finally:
            if os.path.exists(diagdir):
                shutil.rmtree(diagdir)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_avgminmaxer(self):
        """MinMaxAvg should run and produce valid statistics."""
        from mpisppy.extensions.avgminmaxer import MinMaxAvg
        options = self._copy_options()
        options["PHIterLimit"] = 3
        options["avgminmax_name"] = "FirstStageCost"
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            extensions=MinMaxAvg,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)

    def _has_ph_objective_terms(self, s):
        """Return (has_W, has_prox) for a scenario's PH objective terms."""
        m = s._mpisppy_model
        return hasattr(m, "WExpr"), hasattr(m, "ProxExpr")

    def _objective_has_quadratic(self, s):
        """True if the active objective has any quadratic/nonlinear part."""
        from pyomo.repn import generate_standard_repn
        import mpisppy.utils.sputils as sputils
        obj = sputils.find_active_objective(s)
        repn = generate_standard_repn(obj.expr, compute_values=False,
                                      quadratic=True)
        return bool(repn.quadratic_vars) or (repn.nonlinear_expr is not None)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_iter0_uses_user_objective(self):
        """W/prox terms are absent during iteration 0 and attached afterward.

        Regression test for issue #772: iteration 0 should solve exactly the
        objective the user passed in -- no PH machinery (W, prox, xsqvar) in
        the expression tree -- with the terms spliced in only at the end of
        Iter0.
        """
        ph = mpisppy.opt.ph.PH(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        ph.PH_Prep(attach_prox=True)
        s0 = ph.local_scenarios[self.scenario_names[0]]

        # Before Iter0: pure user objective, no PH terms, not quadratic.
        has_W, has_prox = self._has_ph_objective_terms(s0)
        self.assertFalse(has_W, "W term should be absent before Iter0")
        self.assertFalse(has_prox, "prox term should be absent before Iter0")
        self.assertFalse(self._objective_has_quadratic(s0),
                         "iteration-0 objective should be the user's "
                         "(non-quadratic for farmer) objective")

        ph.Iter0()

        # After Iter0: PH terms attached and re-enabled.
        has_W, has_prox = self._has_ph_objective_terms(s0)
        self.assertTrue(has_W, "W term should be attached after Iter0")
        self.assertTrue(has_prox, "prox term should be attached after Iter0")
        self.assertFalse(ph.prox_disabled,
                         "prox should be re-enabled after Iter0")
        self.assertFalse(ph.W_disabled,
                         "W should be re-enabled after Iter0")

    @unittest.skipIf(not persistent_available,
                     "%s solver is not available" % (persistent_solver_name,))
    def test_iter0_defers_with_prox_approx_persistent(self):
        """Deferred attach works with prox linearization on a persistent solver.

        With ``linearize_proximal_terms`` the prox term introduces the new
        ``xsqvar`` variable and cut constraints inside attach_PH_to_objective.
        Because the persistent solver was built on the user's objective during
        Iter0, the deferred attach must re-push the instance. Confirms xsqvar
        is absent before Iter0, present after, and the run completes.
        """
        options = self._copy_options()
        options["solver_name"] = persistent_solver_name
        options["PHIterLimit"] = 20
        options["linearize_proximal_terms"] = True
        options["proximal_linearization_tolerance"] = 1e-1
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        ph.PH_Prep(attach_prox=True)
        s0 = ph.local_scenarios[self.scenario_names[0]]
        self.assertFalse(hasattr(s0._mpisppy_model, "xsqvar"),
                         "xsqvar should not exist before Iter0")

        ph.Iter0()
        self.assertTrue(hasattr(s0._mpisppy_model, "xsqvar"),
                        "xsqvar should be created by the deferred attach")

        # Remaining iterations exercise the re-pushed persistent instance.
        ph.iterk_loop()
        obj = ph.post_loops(ph.extensions)
        self.assertAlmostEqual(obj, FARMER_EF_OBJ, delta=500)


class TestPHMainSizes(unittest.TestCase):
    """Test PH.ph_main() with the sizes (MIP) model, including fixer."""

    def setUp(self):
        self.options = {
            "solver_name": solver_name,
            "PHIterLimit": 10,
            "defaultPHrho": 1,
            "convthresh": 0.001,
            "verbose": False,
            "display_timing": False,
            "display_progress": False,
            "iter0_solver_options": {"mipgap": 0.1, "threads": 1},
            "iterk_solver_options": {"mipgap": 0.02, "threads": 1},
            "smoothed": 0,
            "asynchronousPH": False,
            "toc": False,
        }
        self.scenario_names = [f"Scenario{i+1}" for i in range(3)]
        self.creator_kwargs = {"scenario_count": 3}

    def _copy_options(self):
        return dict(self.options)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_sizes_obj_range(self):
        """PH on sizes should produce an objective in a reasonable range."""
        ph = mpisppy.opt.ph.PH(
            self._copy_options(),
            self.scenario_names,
            sizes_creator,
            sizes_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)
        # sizes optimal is around 227000; PH obj includes prox terms
        self.assertGreater(obj, 100000)
        self.assertLess(obj, 400000)
        # trivial bound should be less than the PH obj
        self.assertLess(tbound, obj)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_sizes_frac_converger_exercises_ints(self):
        """FractionalConverger on sizes actually counts integer variables."""
        from mpisppy.convergers.fracintsnotconv import FractionalConverger
        options = self._copy_options()
        options["PHIterLimit"] = 5
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            sizes_creator,
            sizes_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            ph_converger=FractionalConverger,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)
        # The converger should have run; conv is the fraction not converged
        # (might or might not have reached convergence in 5 iters)
        self.assertIsNotNone(conv)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_sizes_fixer(self):
        """Fixer extension should fix some integer variables."""
        from mpisppy.extensions.fixer import Fixer
        options = self._copy_options()
        options["PHIterLimit"] = 10
        options["fixeroptions"] = {
            "id_fix_list_fct": id_fix_list_fct,
            "verbose": False,
            "boundtol": 0.01,
        }
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            sizes_creator,
            sizes_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            extensions=Fixer,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)
        self.assertGreater(obj, 100000)
        self.assertLess(obj, 400000)
        # Check that at least some variables got fixed
        total_fixed = 0
        for sname, s in ph.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.is_fixed():
                    total_fixed += 1
        # With 3 scenarios and the sizes fixer settings, we expect some fixing
        self.assertGreater(total_fixed, 0,
                           "Fixer should have fixed at least one variable")


class TestPHPrepIsIdempotent(unittest.TestCase):
    """ PH_Prep may be re-entered (a second ph_main() on the same object).

        It used to re-declare W/rho -- silently discarding whatever was in
        them -- and append a second PH term to the already-augmented saved
        objective, leaving the first term live and anchored to the stale xbar.
    """

    def setUp(self):
        self.options = {
            "solver_name": solver_name,
            "PHIterLimit": 50,
            "defaultPHrho": 1,
            "convthresh": 1e-8,
            "verbose": False,
            "display_timing": False,
            "display_progress": False,
            "smoothed": 0,
            "toc": False,
        }
        self.scenario_names = [f"Scenario{i+1}" for i in range(3)]

    def _make_ph(self):
        return mpisppy.opt.ph.PH(
            dict(self.options), self.scenario_names, farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs={"crops_multiplier": 1})

    def test_second_prep_keeps_W_and_rho(self):
        ph = self._make_ph()
        ph.PH_Prep()
        marks = {}
        for k, s in ph.local_scenarios.items():
            for idx in s._mpisppy_model.W:
                s._mpisppy_model.W[idx]._value = 7.0
                s._mpisppy_model.rho[idx] = 3.0
            marks[k] = id(s._mpisppy_model.W)
        ph.PH_Prep()
        for k, s in ph.local_scenarios.items():
            self.assertEqual(id(s._mpisppy_model.W), marks[k],
                             "PH_Prep replaced the W Param")
            for idx in s._mpisppy_model.W:
                self.assertEqual(pyo.value(s._mpisppy_model.W[idx]), 7.0)
                self.assertEqual(pyo.value(s._mpisppy_model.rho[idx]), 3.0)

    def test_second_prep_does_not_double_augment_objective(self):
        ph = self._make_ph()
        # defer_attach=False splices the PH terms in during PH_Prep itself,
        # so this needs no solve.
        ph.PH_Prep(defer_attach=False)
        before = {k: str(obj.expr)
                  for k, obj in ph.saved_objectives.items()}
        ph.PH_Prep(defer_attach=False)
        for k, obj in ph.saved_objectives.items():
            self.assertEqual(str(obj.expr), before[k],
                             "a second PH term was spliced into the objective")
            self.assertEqual(str(obj.expr).count("W_on"), 1)

    def test_two_step_attach_still_works(self):
        # FWPH attaches W at PH_Prep and the prox term later, for its LP start
        # (fwph.py). Guarding the objective must not break that: the second
        # call has to add the prox term it asks for.
        ph = self._make_ph()
        ph.PH_Prep(attach_duals=True, attach_prox=False, defer_attach=False)
        after_W = {k: str(o.expr) for k, o in ph.saved_objectives.items()}
        for k, expr in after_W.items():
            self.assertEqual(expr.count("W_on"), 1)
            self.assertEqual(expr.count("prox_on"), 0, "prox attached early")
        ph.attach_PH_to_objective(add_duals=False, add_prox=True)
        for k, o in ph.saved_objectives.items():
            expr = str(o.expr)
            self.assertEqual(expr.count("prox_on"), 1, "prox term not added")
            self.assertEqual(expr.count("W_on"), 1, "W term duplicated")

    def test_repeat_attach_of_the_same_term_is_a_noop(self):
        ph = self._make_ph()
        ph.PH_Prep(defer_attach=False)
        before = {k: str(o.expr) for k, o in ph.saved_objectives.items()}
        ph.attach_PH_to_objective(add_duals=True, add_prox=True)
        for k, o in ph.saved_objectives.items():
            self.assertEqual(str(o.expr), before[k])

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_smoothing_coefficient_does_not_compound(self):
        # Iter0 scales p by rho when smoothed == 2. Preserving components on a
        # re-prep must not preserve an already-scaled p, or run 2 multiplies
        # it by rho again and silently solves a different smoothing problem.
        options = dict(self.options)
        options.update({"PHIterLimit": 3, "defaultPHrho": 5, "smoothed": 2,
                        "defaultPHp": 1, "defaultPHbeta": 0.1,
                        "iter0_solver_options": None,
                        "iterk_solver_options": None})
        ph = mpisppy.opt.ph.PH(
            options, self.scenario_names, farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs={"crops_multiplier": 1})
        s = ph.local_scenarios[self.scenario_names[0]]
        ph.ph_main()
        after_one = [pyo.value(s._mpisppy_model.p[k])
                     for k in s._mpisppy_model.p]
        ph.ph_main()
        after_two = [pyo.value(s._mpisppy_model.p[k])
                     for k in s._mpisppy_model.p]
        # equal to each other is not enough -- both runs could be wrong by the
        # same factor; p should be defaultPHp scaled by rho exactly once
        expected = options["defaultPHp"] * options["defaultPHrho"]
        self.assertEqual(after_one, [expected] * len(after_one))
        self.assertEqual(after_two, [expected] * len(after_two))

    def test_reprep_is_detected_without_the_attach_flags(self):
        # APH calls PH_Prep with both attach flags False and splices its own
        # terms onto saved_objectives, so re-entry cannot be inferred from
        # what attach_PH_to_objective recorded.
        ph = self._make_ph()
        self.assertFalse(ph._PH_prep_done)
        ph.PH_Prep(attach_duals=False, attach_prox=False)
        self.assertTrue(ph._PH_prep_done, "re-entry would go undetected")
        self.assertFalse(ph._PH_duals_attached)
        self.assertFalse(ph._PH_prox_attached)

    def test_bad_default_rho_is_still_rejected_on_a_reprep(self):
        # rho is preserved across a re-prep, but a bad option must not be
        # silently ignored (issue #560).
        ph = self._make_ph()
        ph.PH_Prep()
        ph.options["defaultPHrho"] = -1
        with self.assertRaises(RuntimeError):
            ph.PH_Prep()

    def test_reprep_with_a_different_term_set_is_refused(self):
        # attach_PH_to_objective only ever adds, and Iter0 re-enables
        # W_on/prox_on at the end, so a narrowed request cannot be honored --
        # ph_main(attach_prox=False) after a plain ph_main() would otherwise
        # silently keep solving with the prox term.
        ph = self._make_ph()
        ph.PH_Prep()
        with self.assertRaises(RuntimeError):
            ph.PH_Prep(attach_prox=False)
        # and the same request is still fine
        ph.PH_Prep()

    def test_reprep_with_changed_smoothing_is_refused(self):
        # add_smooth is not part of what attach_PH_to_objective dedups on, so
        # turning smoothing on or off between runs would be a silent no-op in
        # both directions.
        ph = self._make_ph()
        ph.PH_Prep(attach_smooth=0)
        with self.assertRaises(RuntimeError):
            ph.PH_Prep(attach_smooth=2)

    def test_attach_reports_whether_it_changed_anything(self):
        ph = self._make_ph()
        self.assertIs(ph.attach_PH_to_objective(False, False), False)
        ph.PH_Prep(defer_attach=False)
        # everything already spliced in, so a repeat attaches nothing
        self.assertIs(ph.attach_PH_to_objective(True, True), False)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_fwph_lp_start_end_to_end(self):
        # FWPH's LP start is the real consumer of the two-step attach, and
        # nothing else in the suite runs it: the branch needs
        # FW_LP_start_iterations > 0 and the only other test setting the
        # option uses 0. Run it for real.
        #
        # Note this passes on unfixed mpi-sppy too: FWPH preps once, so the
        # double-augmentation bug never bites it. What it guards is the
        # reverse -- that guarding PH_Prep does not break FWPH, which an
        # all-or-nothing signature check on the attach flags does.
        from mpisppy.opt.fwph import FWPH
        import mpisppy.utils.sputils as sputils

        kwargs = {"crops_multiplier": 1}
        ef = sputils.create_EF(self.scenario_names, farmer.scenario_creator,
                               scenario_creator_kwargs=kwargs)
        solver = pyo.SolverFactory(solver_name)
        if '_persistent' in solver_name:
            solver.set_instance(ef)
        solver.solve(ef)
        ef_opt = pyo.value(ef.EF_Obj)

        options = dict(self.options)
        options.update({
            "PHIterLimit": 10, "convthresh": 1e-6,
            "FW_iter_limit": 20, "FW_weight": 0.0, "FW_conv_thresh": 1e-5,
            "stop_check_tol": 1e-5, "FW_LP_start_iterations": 3,
            "FW_verbose": False, "mip_solver_options": {},
            "qp_solver_options": {}, "iter0_solver_options": None,
            "iterk_solver_options": None,
        })
        fw = FWPH(options, self.scenario_names, farmer.scenario_creator,
                  farmer.scenario_denouement,
                  scenario_creator_kwargs=kwargs)
        fw.fwph_main()

        # the LP start attaches prox on top of the W term PH_Prep attached;
        # each has to appear exactly once
        for sname in self.scenario_names:
            expr = str(fw.saved_objectives[sname].expr)
            self.assertEqual(expr.count("W_on"), 1, "W term duplicated")
            self.assertEqual(expr.count("prox_on"), 1,
                             "prox term missing or duplicated")
        # and the run has to produce a valid outer bound (farmer minimizes)
        self.assertLessEqual(fw.best_bound_obj_val, ef_opt + 1e-6)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_ph_main_twice_matches_ph_main_once(self):
        # Re-running to convergence on an unchanged problem must not move the
        # answer. Note this one passed even before PH_Prep was made
        # idempotent -- on farmer the doubled objective happened not to shift
        # the converged point -- so it guards the property, not the old bug.
        # Both runs have to actually converge for this to mean anything -- at
        # the class-level PHIterLimit neither does, and the two stopping points
        # differ by ~0.2 acres for that reason alone.
        self.options["PHIterLimit"] = 300
        self.options["convthresh"] = 1e-4
        ph = self._make_ph()
        ph.ph_main()
        self.assertLess(ph.conv, self.options["convthresh"],
                        "first run hit the iteration limit; test is vacuous")
        s = ph.local_scenarios[self.scenario_names[0]]
        once = {c: pyo.value(v) for c, v in s.DevotedAcreage.items()}
        ph.ph_main()
        twice = {c: pyo.value(v) for c, v in s.DevotedAcreage.items()}
        self.assertLess(ph.conv, self.options["convthresh"],
                        "second run hit the iteration limit; test is vacuous")
        for c in once:
            self.assertAlmostEqual(once[c], twice[c], places=2)


if __name__ == '__main__':
    unittest.main()
