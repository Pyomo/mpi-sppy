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

import pytest
import pyomo.environ as pyo

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
        self.dill = pytest.importorskip("dill")

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
        rule, varname, typename = found[0]
        self.assertEqual(rule, "bounds_rule")
        self.assertEqual(varname, "cfg")
        self.assertEqual(typename, "Config")

    def test_finds_cfg_captured_by_a_constraint_rule(self):
        found = pickle_bundle.find_undillable_closures(
            self._model_with_cfg_in_constraint_rule())
        self.assertEqual([(r, v) for r, v, _ in found], [("con_rule", "cfg")])

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


if __name__ == '__main__':
    unittest.main()
    
