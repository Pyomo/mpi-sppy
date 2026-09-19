###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""PH_Prep runs once per object, and says so rather than quietly solving
something else.

Running it twice used to discard the duals and double the PH term in the
objective. See issue #848 for what supporting a second run would take.
"""

import unittest

import pyomo.environ as pyo

import mpisppy.opt.ph
import mpisppy.tests.examples.farmer as farmer
from mpisppy.tests.utils import get_solver

solver_available, solver_name, _, _ = get_solver()


def _make_ph(**option_overrides):
    options = {
        "solver_name": solver_name,
        "PHIterLimit": 2,
        "defaultPHrho": 1,
        "convthresh": 1e-8,
        "verbose": False,
        "display_timing": False,
        "display_progress": False,
        "smoothed": 0,
        "toc": False,
    }
    options.update(option_overrides)
    return mpisppy.opt.ph.PH(
        options, [f"Scenario{i+1}" for i in range(3)], farmer.scenario_creator,
        farmer.scenario_denouement,
        scenario_creator_kwargs={"crops_multiplier": 1})


class TestPHPrepRunsOnce(unittest.TestCase):

    def test_a_second_prep_is_refused(self):
        ph = _make_ph()
        ph.PH_Prep()
        with self.assertRaisesRegex(RuntimeError, "already been run"):
            ph.PH_Prep()

    def test_the_refusal_leaves_the_first_prep_alone(self):
        # The point of refusing is that the first run's state survives: a
        # second prep used to replace the W Param (dropping whatever was in
        # it) and splice a second PH term into the same objective.
        ph = _make_ph()
        ph.PH_Prep(defer_attach=False)
        marks, expressions = {}, {}
        for k, s in ph.local_scenarios.items():
            for idx in s._mpisppy_model.W:
                s._mpisppy_model.W[idx]._value = 7.0
            marks[k] = id(s._mpisppy_model.W)
            expressions[k] = str(ph.saved_objectives[k].expr)

        with self.assertRaises(RuntimeError):
            ph.PH_Prep(defer_attach=False)

        for k, s in ph.local_scenarios.items():
            self.assertEqual(id(s._mpisppy_model.W), marks[k],
                             "the W Param was replaced")
            for idx in s._mpisppy_model.W:
                self.assertEqual(pyo.value(s._mpisppy_model.W[idx]), 7.0)
            self.assertEqual(str(ph.saved_objectives[k].expr), expressions[k],
                             "a second PH term was spliced in")
            self.assertEqual(expressions[k].count("W_on"), 1)

    def test_a_prep_that_raised_is_not_counted(self):
        # The flag is set once the prep has been through, so a prep that died
        # partway does not leave the object claiming one. Retrying meets the
        # real failure rather than "already been run", which would point at
        # the wrong fix.
        ph = _make_ph(defaultPHrho=-1)      # attach_Ws_and_prox rejects this
        with self.assertRaisesRegex(RuntimeError, "defaultPHrho"):
            ph.PH_Prep()
        self.assertFalse(ph._PH_prep_done)
        ph.options["defaultPHrho"] = 1
        ph.PH_Prep()                        # the corrected call goes through

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_a_second_ph_main_is_refused(self):
        # ph_main preps, so this is the same refusal reached the way a user
        # would reach it.
        ph = _make_ph()
        ph.ph_main()
        with self.assertRaisesRegex(RuntimeError, "already been run"):
            ph.ph_main()

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_one_run_still_works(self):
        ph = _make_ph(PHIterLimit=5)
        conv, obj, trivial_bound = ph.ph_main()
        self.assertIsNotNone(obj)
        # farmer minimizes, so the bound with nonanticipativity dropped is
        # below the expected value of the objective
        self.assertLessEqual(trivial_bound, obj + 1e-6)


if __name__ == "__main__":
    unittest.main()
