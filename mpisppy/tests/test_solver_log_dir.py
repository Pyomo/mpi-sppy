###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""solver_log_dir with each kind of Pyomo solver interface (issue #861).

The solves use highs, a pyomo.contrib.solver interface: those reject the
legacy logfile keyword, so solver_log_dir has to reach them through tee.
These tests require highspy and fail rather than skip without it; CI
installs it for every job that runs this file.
"""

import os
import shutil
import sys
import tempfile
import unittest

import pyomo.environ as pyo

import mpisppy.opt.ph
import mpisppy.tests.examples.farmer as farmer
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm


def _ph_options(solver_name, log_dir):
    return {
        "asynchronousPH": False,
        "solver_name": solver_name,
        "PHIterLimit": 1,
        "defaultPHrho": 1,
        "convthresh": 0.001,
        "verbose": False,
        "display_timing": False,
        "display_progress": False,
        "solver_log_dir": log_dir,
    }


class Test_solver_log_dir(unittest.TestCase):

    def setUp(self):
        self.snames = ["scen0", "scen1", "scen2"]
        self.sck = {"num_scens": 3}
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def _log_dir(self, name):
        # not created here: the constructors refuse an existing directory
        return os.path.join(self.tmpdir, name)

    def _require_highs(self):
        if not pyo.SolverFactory("highs").available(exception_flag=False):
            self.fail("test_solver_log_dir.py requires highspy "
                      "(pip install highspy)")

    def test_log_stream_adds_file_to_tee_and_restores_it(self):
        log_path = os.path.join(self.tmpdir, "stream.log")
        for tee, expected_extra in ((False, []), (True, [sys.stdout])):
            kwargs = {"tee": tee}
            with sputils.solver_log_stream(log_path, kwargs):
                streams = kwargs["tee"]
                self.assertEqual(streams[1:], expected_extra)
                streams[0].write(f"tee={tee}\n")
            self.assertIs(kwargs["tee"], tee)
        kwargs = {}
        with sputils.solver_log_stream(log_path, kwargs):
            self.assertEqual(len(kwargs["tee"]), 1)
        self.assertNotIn("tee", kwargs)
        # appended, so a retry under the same name keeps the first attempt
        with open(log_path) as f:
            self.assertEqual(f.read(), "tee=False\ntee=True\n")

    def test_log_stream_is_a_no_op_without_a_path(self):
        kwargs = {"tee": True}
        with sputils.solver_log_stream(None, kwargs):
            self.assertIs(kwargs["tee"], True)

    def test_set_solver_log_file_dispatch(self):
        log_path = os.path.join(self.tmpdir, "x.log")
        cases = {
            # name: (returns a stream path, sets the logfile keyword)
            "highs": (True, False),
            "gurobi_persistent_v2": (True, False),
            "cplex": (False, True),
        }
        for name, (wants_stream, wants_keyword) in cases.items():
            with self.subTest(solver=name):
                kwargs = {}
                got = sputils.set_solver_log_file(
                    pyo.SolverFactory(name), name, log_path, kwargs)
                self.assertEqual(got == log_path, wants_stream)
                self.assertEqual("logfile" in kwargs, wants_keyword)

        # a pyomo.contrib.solver interface with its own logfile option: it
        # reduces tee to a bool, so a stream in tee would stay empty
        gams = pyo.SolverFactory("gams_v2")
        kwargs = {}
        self.assertIsNone(sputils.set_solver_log_file(
            gams, "gams_v2", log_path, kwargs))
        self.assertEqual(str(gams.config.logfile), log_path)
        self.assertEqual(kwargs, {})

        gurobi = pyo.SolverFactory("gurobi_persistent")
        kwargs = {}
        self.assertIsNone(sputils.set_solver_log_file(
            gurobi, "gurobi_persistent", log_path, kwargs))
        self.assertEqual(gurobi.options["LogFile"], log_path)
        self.assertEqual(kwargs, {})

        appsi_highs = pyo.SolverFactory("appsi_highs")
        appsi_ipopt = pyo.SolverFactory("appsi_ipopt")
        self.assertIsNone(sputils.set_solver_log_file(
            appsi_highs, "appsi_highs", log_path, kwargs))
        self.assertEqual(appsi_highs.config.logfile, log_path)
        self.assertEqual(kwargs, {})
        with self.assertRaises(ValueError):
            sputils.set_solver_log_file(
                appsi_ipopt, "appsi_ipopt", log_path, kwargs)

    def test_agnostic_solve_hands_the_guest_the_log_path(self):
        # With a guest doing the solve, the host plugin's log mechanism is
        # irrelevant: an APPSI host solver with no log file option must not
        # make the run fail, and the guest gets the path as before.
        host_solver = pyo.SolverFactory("appsi_ipopt")

        class RecordingGuest:
            def callout_agnostic(self, kws):
                self.kws = kws

        log_dir = self._log_dir("agnostic")
        ph = mpisppy.opt.ph.PH(
            _ph_options("appsi_ipopt", log_dir),
            self.snames,
            farmer.scenario_creator,
            scenario_creator_kwargs=self.sck,
        )
        ph.Ag = RecordingGuest()
        k, s = next(iter(ph.local_scenarios.items()))
        s._solver_plugin = host_solver
        ph.solve_one(None, k, s)
        self.assertEqual(
            ph.Ag.kws["solve_keyword_args"]["logfile"],
            os.path.join(log_dir, f"{ph._subproblem_file_stem(k)}_0.log"))

    def test_contrib_solver_ef_writes_log(self):
        self._require_highs()
        log_dir = self._log_dir("ef")
        ef = ExtensiveForm(
            options={"solver": "highs", "solver_log_dir": log_dir},
            all_scenario_names=self.snames,
            scenario_creator=farmer.scenario_creator,
            scenario_creator_kwargs=self.sck)
        results = ef.solve_extensive_form()
        pyo.assert_optimal_termination(results)
        log_file = os.path.join(log_dir, "EF_solver_log.log")
        self.assertGreater(os.path.getsize(log_file), 0)

    def test_contrib_solver_ph_writes_log(self):
        self._require_highs()
        log_dir = self._log_dir("ph")
        ph = mpisppy.opt.ph.PH(
            _ph_options("highs", log_dir),
            self.snames,
            farmer.scenario_creator,
            scenario_creator_kwargs=self.sck,
        )
        ph.ph_main()
        logs = sorted(os.listdir(log_dir))
        # iter0 and iter1 for each scenario
        self.assertEqual(len(logs), 2 * len(self.snames), logs)
        for log in logs:
            self.assertGreater(
                os.path.getsize(os.path.join(log_dir, log)), 0, log)


if __name__ == '__main__':
    unittest.main()
