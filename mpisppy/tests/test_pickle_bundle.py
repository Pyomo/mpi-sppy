###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Provide a test for special-purpose (aircond only)multi-stage pickled bundles
"""

"""

import os
import sys
import tempfile
import unittest

import pyomo.environ as pyo
from pyomo.common.dependencies import DeferredImportError

import mpisppy.utils.pickle_bundle as pickle_bundle
from mpisppy.utils.config import Config

import mpisppy.MPI as mpi

from mpisppy.tests.utils import get_solver

# Parse --python-args (extra args inserted after "python" in subcommands, e.g. for coverage)
python_args = ""
_remaining = []
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i].startswith("--python-args="):
        python_args = sys.argv[_i].split("=", 1)[1]
    elif sys.argv[_i] == "--python-args" and _i + 1 < len(sys.argv):
        _i += 1
        python_args = sys.argv[_i]
    else:
        _remaining.append(sys.argv[_i])
    _i += 1
sys.argv = [sys.argv[0]] + _remaining


fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

__version__ = 0.1

solver_available, solver_name, persistent_available, persistent_solver_name= get_solver()
module_dir = os.path.dirname(os.path.abspath(__file__))

badguys = list()

#*****************************************************************************
class Test_pickle_bundles(unittest.TestCase):
    """ Test the pickle bundle code using aircond."""

    @classmethod
    def setUpClass(self):
        self.refmodelname ="mpisppy.tests.examples.aircond"  # amalgamator compatible
        self.aircond_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples', 'aircond')

        self.BF1 = 2
        self.BF2 = 2
        self.BF3 = 2
        self.SPB = self.BF2 * self.BF3  # implies bundle count
        self.SC = self.BF1 * self.BF2 * self.BF3
        self.BPF = self.BF1  # bundle count (by design)
        self.BF_str = f"--branching-factors \"{self.BF1} {self.BF2} {self.BF3}\""

        self.BI=150
        self.NC=1
        self.QSC=0.3
        self.SD=80
        self.OTC=1.5
        self.EC = f"--Capacity 200 --QuadShortCoeff {self.QSC}  --BeginInventory {self.BI} --mu-dev 0 --sigma-dev {self.SD} --start-seed 0 --NegInventoryCost={self.NC} --OvertimeProdCost={self.OTC}"

    def setUp(self):
        self.cwd = os.getcwd()
        self.tempdir = tempfile.TemporaryDirectory()
        self.tempdir_name = self.tempdir.name
        os.chdir(self.aircond_dir)
        
    def tearDown(self):
        os.chdir(self.cwd)


    def test_pickle_bundler(self):
        cmdstr = f"python {python_args} bundle_pickler.py {self.BF_str} --pickle-bundles-dir={self.tempdir_name} --scenarios-per-bundle={self.SPB} {self.EC}"
        ret = os.system(cmdstr)
        if ret != 0:
            raise RuntimeError(f"Test run failed with code {ret}")

    def test_chain(self):
        # run the pickle bundler then aircond_cylinders
        cmdstr = f"python {python_args} bundle_pickler.py {self.BF_str} --pickle-bundles-dir={self.tempdir_name} --scenarios-per-bundle={self.SPB} {self.EC}"
        ret = os.system(cmdstr)
        if ret != 0:
            raise RuntimeError(f"pickler part of test run failed with code {ret}")

        cmdstr = f"python {python_args} aircond_cylinders.py --branching-factors=\"{self.BPF}\" --unpickle-bundles-dir={self.tempdir_name} --scenarios-per-bundle={self.SPB} {self.EC} "+\
                 f"--default-rho=1 --max-solver-threads=2 --bundles-per-rank=0 --max-iterations=2 --solver-name={solver_name}"
        ret = os.system(cmdstr)
        if ret != 0:
            raise RuntimeError(f"cylinders part of test run failed with code {ret}")

@unittest.skipUnless(pickle_bundle.dill_available,
                     "dill is not installed")
class Test_dill_failure_diagnostics(unittest.TestCase):
    """A model that cannot be dilled must say *why*, not just that it failed.

    dill's own message is typically opaque -- "args[0] from __newobj__ args has
    the wrong class" -- and names neither the component nor the offending
    object. The common cause is a value captured in the closure of a Pyomo
    rule: Pyomo keeps the rule function on the component it built, so whatever
    the rule closed over has to be serializable too. These tests pin that the
    diagnostic finds such a value and names it, and -- just as important --
    that it does not claim that cause when the real problem is something else.
    """

    def setUp(self):
        import dill
        self.dill = dill

    @staticmethod
    def _model_with_cfg_in_bounds_rule():
        # Rules must be nested to create a real closure, as they are inside a
        # scenario_creator; a module-level rule captures a global instead.
        cfg = Config()
        cfg.popular_args()
        model = pyo.ConcreteModel()

        def bounds_rule(m, i):
            return (0, 10 if cfg.max_iterations else 5)

        model.v = pyo.Var([1, 2], bounds=bounds_rule)
        return model

    @staticmethod
    def _model_with_cfg_in_constraint_rule():
        cfg = Config()
        cfg.popular_args()
        model = pyo.ConcreteModel()
        model.x = pyo.Var([1, 2])

        def con_rule(m, i):
            return m.x[i] >= (1 if cfg.max_iterations else 0)

        model.c = pyo.Constraint([1, 2], rule=con_rule)
        return model

    @staticmethod
    def _clean_model():
        model = pyo.ConcreteModel()
        limit = 10

        def bounds_rule(m, i):
            return (0, limit)

        model.v = pyo.Var([1, 2], bounds=bounds_rule)
        return model

    def test_finds_cfg_captured_by_a_var_bounds_rule(self):
        """A Var bounds rule sits three levels down in Pyomo's initializers."""
        found = pickle_bundle.find_undillable_closures(
            self._model_with_cfg_in_bounds_rule())
        self.assertEqual(len(found), 1)
        rule, varname, typename, kind = found[0]
        self.assertEqual(rule, "bounds_rule")
        self.assertEqual(varname, "cfg")
        self.assertEqual(typename, "Config")
        self.assertEqual(kind, "closure")

    def test_finds_cfg_captured_by_a_constraint_rule(self):
        found = pickle_bundle.find_undillable_closures(
            self._model_with_cfg_in_constraint_rule())
        self.assertEqual([(r, v) for r, v, _, _ in found],
                         [("con_rule", "cfg")])

    def test_clean_model_has_no_findings(self):
        self.assertEqual(
            pickle_bundle.find_undillable_closures(self._clean_model()), [])

    def test_description_names_the_rule_and_the_variable(self):
        model = self._model_with_cfg_in_bounds_rule()
        with self.assertRaises(Exception) as ctx:
            self.dill.dumps(model)
        msg = pickle_bundle.describe_dill_failure(model, ctx.exception)
        self.assertIn("bounds_rule", msg)
        self.assertIn("cfg", msg)
        self.assertIn("Config", msg)

    def test_description_does_not_invent_a_closure_cause(self):
        """The failure mode that matters: do not misdiagnose other causes.

        A model that fails for an unrelated reason must not be handed a
        confident explanation about rule closures that does not apply to it.
        """
        model = pyo.ConcreteModel()
        model.x = pyo.Var([1, 2])
        model._gen = (i for i in range(3))   # generators cannot be pickled
        with self.assertRaises(Exception) as ctx:
            self.dill.dumps(model)
        msg = pickle_bundle.describe_dill_failure(model, ctx.exception)
        self.assertIn("No unserializable value was found", msg)
        self.assertNotIn("closes over", msg)
        # The underlying error is still reported verbatim.
        self.assertIn("generator", msg)

    def test_dill_pickle_raises_with_the_diagnosis(self):
        model = self._model_with_cfg_in_bounds_rule()
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "scen.dill")
            with self.assertRaises(RuntimeError) as ctx:
                pickle_bundle.dill_pickle(model, fname)
            self.assertIn("bounds_rule", str(ctx.exception))
            self.assertFalse(
                os.path.exists(fname),
                msg="a truncated pickle was left behind for a later run to "
                    "try to load")

    def test_dill_pickle_still_works_on_a_good_model(self):
        model = self._clean_model()
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "scen.dill")
            pickle_bundle.dill_pickle(model, fname)
            reloaded = pickle_bundle.dill_unpickle(fname)
            self.assertEqual(len(list(reloaded.v)), 2)

    def test_dill_unpickle_explains_an_unreadable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "truncated.dill")
            with open(fname, "wb") as f:
                f.write(b"not a pickle")
            with self.assertRaises(RuntimeError) as ctx:
                pickle_bundle.dill_unpickle(fname)
            self.assertIn("truncated.dill", str(ctx.exception))

    def test_io_errors_are_not_dressed_up_as_serialization_failures(self):
        """A bad path speaks for itself; do not lecture about rule closures."""
        model = self._clean_model()
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "no_such_dir", "scen.dill")
            with self.assertRaises(OSError):
                pickle_bundle.dill_pickle(model, missing)
        with self.assertRaises(OSError):
            pickle_bundle.dill_unpickle(
                os.path.join(tempfile.gettempdir(), "mpisppy_no_such_file.dill"))

    def test_failed_open_does_not_delete_a_pre_existing_file(self):
        """The cleanup must only remove a file this call actually wrote."""
        model = self._model_with_cfg_in_bounds_rule()
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "precious.dill")
            with open(fname, "wb") as f:
                f.write(b"someone else's bytes")
            os.chmod(fname, 0o400)          # unwritable file, writable dir
            try:
                with self.assertRaises(OSError):
                    pickle_bundle.dill_pickle(model, fname)
                self.assertTrue(
                    os.path.exists(fname),
                    msg="a file this call never wrote to was deleted")
            finally:
                os.chmod(fname, 0o600)

    def test_write_error_does_not_leave_a_truncated_pickle(self):
        """A write failure (disk full, quota) must not leave a partial file.

        open() succeeded and truncated the target, so what is on disk is our
        own partial write -- exactly the thing a later --unpickle-* run would
        try to load.
        """
        class _FailingDill:
            def dump(self, model, f):
                f.write(b"x" * 200)
                raise OSError(28, "No space left on device")

        model = self._clean_model()
        saved = pickle_bundle.dill
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "scen.dill")
            try:
                pickle_bundle.dill = _FailingDill()
                with self.assertRaises(OSError):
                    pickle_bundle.dill_pickle(model, fname)
            finally:
                pickle_bundle.dill = saved
            self.assertFalse(
                os.path.exists(fname),
                msg="a truncated pickle survived a write failure")

    def test_missing_dill_does_not_truncate_the_target(self):
        """Refuse before opening, or we destroy a file we cannot rewrite."""
        model = self._clean_model()
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "existing.dill")
            with open(fname, "wb") as f:
                f.write(b"previous contents")
            saved = pickle_bundle.dill_available
            try:
                pickle_bundle.dill_available = False
                with self.assertRaises(RuntimeError) as ctx:
                    pickle_bundle.dill_pickle(model, fname)
            finally:
                pickle_bundle.dill_available = saved
            self.assertIn("dill is required", str(ctx.exception))
            with open(fname, "rb") as f:
                self.assertEqual(f.read(), b"previous contents")

    def test_large_bound_method_owner_is_not_expanded(self):
        """A Pyomo component owner would bury the answer in red herrings."""
        cfg = Config()
        cfg.popular_args()
        model = pyo.ConcreteModel()
        model.x = pyo.Var([1, 2])

        def con_rule(m, i):
            return m.x[i] >= (1 if cfg.max_iterations else 0)

        model.c = pyo.Constraint([1, 2], rule=con_rule)
        model._helper = model.clone       # bound method owned by the model

        found = pickle_bundle.find_undillable_closures(model)
        self.assertTrue(found)
        self.assertEqual(
            [varname for _, varname, _, kind in found if kind == "self"], [],
            msg="expanded a Pyomo component as if it were a model builder")

    def test_duplicate_captures_are_reported_once(self):
        """A bundle carries one copy of each rule per scenario."""
        def build_block(cfg):
            blk = pyo.Block(concrete=True)
            blk.x = pyo.Var([1, 2])

            def con_rule(b, i):
                return b.x[i] >= (1 if cfg.max_iterations else 0)

            blk.c = pyo.Constraint([1, 2], rule=con_rule)
            return blk

        cfg = Config()
        cfg.popular_args()
        model = pyo.ConcreteModel()
        for i in range(5):                  # five "scenarios", one shared cfg
            setattr(model, f"s{i}", build_block(cfg))

        found = pickle_bundle.find_undillable_closures(model)
        self.assertEqual(
            found, [("con_rule", "cfg", "Config", "closure")],
            msg="the same capture was reported once per scenario")

    def test_bound_method_rule_is_diagnosed(self):
        """A class-based builder reaches its config through self, not a cell."""
        class Builder:
            def __init__(self):
                self.cfg = Config()
                self.cfg.popular_args()

            def con_rule(self, m, i):
                return m.x[i] >= (1 if self.cfg.max_iterations else 0)

        builder = Builder()
        model = pyo.ConcreteModel()
        model.x = pyo.Var([1, 2])
        model.c = pyo.Constraint([1, 2], rule=builder.con_rule)

        with self.assertRaises(Exception):
            self.dill.dumps(model)
        found = pickle_bundle.find_undillable_closures(model)
        self.assertTrue(found, msg="bound-method rule was not diagnosed")
        self.assertIn("self.cfg", [varname for _, varname, _, _ in found])


class Test_dill_diagnostics_without_dill(unittest.TestCase):
    """With dill absent the diagnostic must not invent a culprit.

    dill is optional and reached through Pyomo's attempt_import, so calling it
    raises DeferredImportError -- an ordinary Exception. Without a guard, the
    serializability probe treats every closure value as unserializable and
    tells the user to rewrite a scenario_creator over, say, an int.
    """

    def test_find_returns_nothing_when_dill_is_missing(self):
        """Simulate a genuinely absent dill, not just a lowered flag.

        attempt_import hands back a module proxy that raises on first use, so
        every serializability probe raises rather than returning False. Merely
        setting dill_available = False while dill is still importable would not
        reproduce the bug this guards.
        """
        model = pyo.ConcreteModel()
        limit = 10                      # perfectly serializable

        def bounds_rule(m, i):
            return (0, limit)

        model.v = pyo.Var([1, 2], bounds=bounds_rule)

        class _AbsentDill:
            def dumps(self, *args, **kwargs):
                raise DeferredImportError(
                    "The dill module (an optional mpi-sppy dependency) failed "
                    "to import: No module named 'dill'")

        saved_flag = pickle_bundle.dill_available
        saved_mod = pickle_bundle.dill
        try:
            pickle_bundle.dill_available = False
            pickle_bundle.dill = _AbsentDill()
            self.assertEqual(
                pickle_bundle.find_undillable_closures(model), [],
                msg="with dill absent, every closure value was misreported "
                    "as the culprit")
        finally:
            pickle_bundle.dill_available = saved_flag
            pickle_bundle.dill = saved_mod

    def test_description_says_dill_is_missing(self):
        model = pyo.ConcreteModel()
        saved = pickle_bundle.dill_available
        try:
            pickle_bundle.dill_available = False
            msg = pickle_bundle.describe_dill_failure(
                model, RuntimeError("boom"))
        finally:
            pickle_bundle.dill_available = saved
        self.assertIn("dill is not installed", msg)
        self.assertNotIn("closes over", msg)

    def test_broken_diagnostic_is_disclosed_not_hidden(self):
        """If the walk itself breaks, say so instead of claiming it ran clean."""
        model = pyo.ConcreteModel()
        saved = pickle_bundle.find_undillable_closures

        def boom(_model):
            raise TypeError("diagnostic itself is broken")

        try:
            pickle_bundle.find_undillable_closures = boom
            msg = pickle_bundle.describe_dill_failure(
                model, RuntimeError("original failure"))
        finally:
            pickle_bundle.find_undillable_closures = saved
        self.assertIn("closure diagnostic itself failed", msg)
        self.assertIn("diagnostic itself is broken", msg)


if __name__ == '__main__':
    unittest.main()
    
