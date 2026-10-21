###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Author:
"""
IMPORTANT:
  Unless we run to convergence, the solver, and even solver
version matter a lot, so we often just do smoke tests.
"""

import csv
import os
import tempfile
import unittest
from unittest import mock

import pyomo.environ as pyo

import mpisppy.opt.ph
import mpisppy.tests.examples.farmer as farmer
import mpisppy.tests.examples.hydro.hydro as hydro
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.w_utils.wxbarreader as wxbarreader
import mpisppy.utils.w_utils.wxbarutils as wxbarutils
import mpisppy.utils.w_utils.wxbarwriter as wxbarwriter
from mpisppy.extensions.extension import Extension, MultiExtension
from mpisppy.extensions.sep_rho import SepRho
from mpisppy.phbase import PHBase
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver
from mpisppy.utils import config, sputils

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_W_TEST_DATA_DIR = os.path.join(_THIS_DIR, "examples", "w_test_data")

__version__ = 0.2

solver_available,solver_name, persistent_available, persistent_solver_name= get_solver()

def _create_cfg():
    cfg = config.Config()
    wxbarreader.add_options_to_config(cfg)        
    wxbarwriter.add_options_to_config(cfg)
    cfg.add_branching_factors()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.solver_name = solver_name
    cfg.default_rho = 1
    cfg.max_solver_threads = 1
    return cfg

class _CostCoeffRecorder(Extension):
    """ Records the nonant cost coefficients in pre_iter0, as SepRho and
    CoeffRho read them there."""
    def pre_iter0(self):
        self.coeffs = {sname: sputils.nonant_cost_coeffs(s)
                       for sname, s in self.opt.local_scenarios.items()}

#*****************************************************************************

class Test_xbar_w_reader_writer_farmer(unittest.TestCase):
    """ Test the gradient code using farmer."""

    def _create_ph_farmer(self, ph_extensions=None, max_iter=100,
                          extension_kwargs=None, read_xbar=True):
        self.w_file_name = os.path.join(_W_TEST_DATA_DIR, 'w_file.csv')
        self.temp_w_file_name = os.path.join(_W_TEST_DATA_DIR, '_temp_w_file.csv')
        self.xbar_file_name = os.path.join(_W_TEST_DATA_DIR, 'xbar_file.csv')
        self.temp_xbar_file_name = os.path.join(_W_TEST_DATA_DIR, '_temp_xbar_file.csv')
        self.cfg.num_scens = 3
        scenario_creator = farmer.scenario_creator
        scenario_denouement = farmer.scenario_denouement
        all_scenario_names = farmer.scenario_names_creator(self.cfg.num_scens)
        scenario_creator_kwargs = farmer.kw_creator(self.cfg)
        self.cfg.max_iterations = max_iter
        beans = (self.cfg, scenario_creator, scenario_denouement, all_scenario_names)
        hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs,
                                  ph_extensions=ph_extensions, extension_kwargs=extension_kwargs)
        hub_dict['opt_kwargs']['options']['cfg'] = self.cfg
        if ph_extensions==wxbarwriter.WXBarWriter:
            self.cfg.W_and_xbar_writer = True
            self.cfg.W_fname = self.temp_w_file_name
            self.cfg.Xbar_fname = self.temp_xbar_file_name
        if ph_extensions==wxbarreader.WXBarReader or \
                wxbarreader.WXBarReader in (extension_kwargs or {}).get("ext_classes", []):
            self.cfg.W_and_xbar_reader = True
            self.cfg.init_W_fname = self.w_file_name
            if read_xbar:
                self.cfg.init_Xbar_fname = self.xbar_file_name
        list_of_spoke_dict = list()
        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()
        if wheel.strata_rank == 0:
            ph_object = wheel.spcomm.opt
            return ph_object

    def setUp(self):
        self.cfg = _create_cfg()
        self.ph_object = None

    def test_wwriter(self):
        self.ph_object = self._create_ph_farmer(ph_extensions=wxbarwriter.WXBarWriter, max_iter=5)
        with open(self.temp_w_file_name) as f:
            read = csv.reader(f)
            rows = list(read)
            self.assertAlmostEqual(float(rows[1][2]), 70.84705093609978, places=5)
            self.assertAlmostEqual(float(rows[3][2]), -41.104251445950844, places=5)
        os.remove(self.temp_w_file_name)

    def test_xbarwriter(self):
        self.ph_object = self._create_ph_farmer(ph_extensions=wxbarwriter.WXBarWriter, max_iter=5)
        with open(self.temp_xbar_file_name) as f:
            read = csv.reader(f)
            rows = list(read)
            self.assertAlmostEqual(float(rows[1][1]), 274.2239371483933, places=5)
            self.assertAlmostEqual(float(rows[3][1]), 96.88717449844287, places=5)
        os.remove(self.temp_xbar_file_name)

    def test_wreader(self):
        self.ph_object = self._create_ph_farmer(ph_extensions=wxbarreader.WXBarReader, max_iter=0)
        for sname, scenario in self.ph_object.local_scenarios.items():
            if sname == 'scen0':
                self.assertAlmostEqual(scenario._mpisppy_model.W[("ROOT", 1)]._value, 70.84705093609978)
            if sname == 'scen1':
                self.assertAlmostEqual(scenario._mpisppy_model.W[("ROOT", 0)]._value, -41.104251445950844)

    def test_xbarreader(self):
        self.ph_object = self._create_ph_farmer(ph_extensions=wxbarreader.WXBarReader, max_iter=0)
        for sname, scenario in self.ph_object.local_scenarios.items():
            if sname == 'scen0':
                self.assertAlmostEqual(scenario._mpisppy_model.xbars[("ROOT", 1)]._value, 274.2239371483933)
            if sname == 'scen1':
                self.assertAlmostEqual(scenario._mpisppy_model.xbars[("ROOT", 0)]._value, 96.88717449844287)


    def _solve_iter0_outside_PH(self, with_xbar):
        # Independent of PH: min_x f_s(x) + W_s x, plus
        # rho/2 (x - xbar)^2 when with_xbar, for each scenario, with W_s and
        # xbar from the files. Returns sum_s p_s of the optimal values and the
        # first-stage solution of each scenario.
        W = {}
        with open(self.w_file_name) as f:
            for sname, vname, val in csv.reader(f):
                W[sname, vname] = float(val)
        xbar = {}
        with open(self.xbar_file_name) as f:
            for vname, val in csv.reader(f):
                xbar[vname] = float(val)
        rho = self.cfg.default_rho
        kwargs = farmer.kw_creator(self.cfg)
        names = farmer.scenario_names_creator(self.cfg.num_scens)
        Eobj = 0.0
        xs = {}
        for sname in names:
            m = farmer.scenario_creator(sname, **kwargs)
            obj = sputils.find_active_objective(m)
            nonants = [v for node in m._mpisppy_node_list
                       for v in node.nonant_vardata_list]
            obj.expr += sum(W[sname, v.name] * v for v in nonants)
            if with_xbar:
                obj.expr += sum(rho / 2.0 * (v - xbar[v.name])**2
                                for v in nonants)
            opt = pyo.SolverFactory(solver_name)
            if sputils.is_persistent(opt):
                opt.set_instance(m)
                opt.solve()
            else:
                opt.solve(m)
            Eobj += m._mpisppy_probability * pyo.value(obj)
            xs[sname] = {v.name: v.value for v in nonants}
        return Eobj, xs

    def test_wreader_iter0_uses_W(self):
        cold = self._create_ph_farmer(max_iter=0).trivial_bound
        self.setUp()
        warm = self._create_ph_farmer(ph_extensions=wxbarreader.WXBarReader,
                                      max_iter=0, read_xbar=False).trivial_bound
        expected, _ = self._solve_iter0_outside_PH(with_xbar=False)
        self.assertGreater(abs(expected - cold), 1.0)
        self.assertAlmostEqual(warm, expected, places=3)

    def test_wxbarreader_iter0_uses_W_and_xbar(self):
        ph = self._create_ph_farmer(ph_extensions=wxbarreader.WXBarReader,
                                    max_iter=0)
        _, expected = self._solve_iter0_outside_PH(with_xbar=True)
        _, without_prox = self._solve_iter0_outside_PH(with_xbar=False)
        # The prox term must move the solution, or matching it shows nothing.
        self.assertGreater(
            max(abs(expected[sname][vname] - without_prox[sname][vname])
                for sname in expected for vname in expected[sname]), 1.0)
        for sname, s in ph.local_scenarios.items():
            for node in s._mpisppy_node_list:
                for v in node.nonant_vardata_list:
                    self.assertAlmostEqual(v.value, expected[sname][v.name],
                                           places=2)

    def test_wxbarreader_with_linearized_prox(self):
        # Before iteration 0 the nonants have no values, so the prox
        # approximation has to start from xbar rather than from a solution.
        self.cfg.linearize_proximal_terms = True
        ph = self._create_ph_farmer(ph_extensions=wxbarreader.WXBarReader,
                                    max_iter=0)
        for s in ph.local_scenarios.values():
            for manager in s._mpisppy_data.xsqvar_prox_approx.values():
                self.assertTrue(
                    any(abs(c - manager.xbar.value) < 1e-9
                        for c in manager.cut_values),
                    f"no cut at xbar for {manager.xvar.name}")

    def test_wreader_with_linearized_prox(self):
        # W only: the prox term is off in iteration 0, and the first cuts
        # come from the iteration-0 solution as on a cold run.
        self.cfg.linearize_proximal_terms = True
        ph = self._create_ph_farmer(ph_extensions=wxbarreader.WXBarReader,
                                    max_iter=2, read_xbar=False)
        self.assertEqual(ph._PHIter, 2)

    def test_xbar_read_gives_no_trivial_bound(self):
        # The prox term is in the iteration-0 objective, so those values are
        # not a bound; W alone still gives one (test_wreader_iter0_uses_W).
        ph = self._create_ph_farmer(ph_extensions=wxbarreader.WXBarReader,
                                    max_iter=0)
        self.assertIsNone(ph.trivial_bound)

    def test_xbar_with_smoothing_is_refused(self):
        self.cfg.smoothing = True
        with self.assertRaisesRegex(RuntimeError, "smoothing"):
            self._create_ph_farmer(ph_extensions=wxbarreader.WXBarReader,
                                   max_iter=0)

    def test_iter0_with_xbar_checks_the_solver_for_quadratics(self):
        # With xbar read, iteration 0 is the first quadratic solve, so the
        # capability check has to run before it, not before iteration 1.
        with mock.patch("mpisppy.utils.sputils.solver_quadratic_objective_capability",
                        return_value=False):
            with self.assertRaisesRegex(RuntimeError, "linearize-proximal-terms"):
                self._create_ph_farmer(ph_extensions=wxbarreader.WXBarReader,
                                       max_iter=0)

    def _iterations_checked_after_solve(self, read_xbar):
        checked = []
        real = PHBase._check_prox_solve_succeeded

        def record(ph):
            checked.append(ph._PHIter)
            return real(ph)
        with mock.patch.object(PHBase, "_check_prox_solve_succeeded",
                               autospec=True, side_effect=record):
            self._create_ph_farmer(ph_extensions=wxbarreader.WXBarReader,
                                   max_iter=0, read_xbar=read_xbar)
        return checked

    def test_iter0_with_xbar_checks_its_solve(self):
        self.assertEqual(self._iterations_checked_after_solve(True), [0])

    def test_iter0_with_W_only_is_not_a_prox_solve(self):
        self.assertEqual(self._iterations_checked_after_solve(False), [])

    def test_prox_approx_refuses_a_nonant_without_value_after_iter0(self):
        self.cfg.linearize_proximal_terms = True
        ph = self._create_ph_farmer(ph_extensions=wxbarreader.WXBarReader,
                                    max_iter=0)
        s = next(iter(ph.local_scenarios.values()))
        next(iter(s._mpisppy_data.nonant_indices.values()))._value = None
        ph._PHIter = 1
        with self.assertRaisesRegex(RuntimeError, "has no value at PH iteration 1"):
            ph._update_prox_approx()

    def test_wreader_leaves_pre_iter0_cost_coeffs_alone(self):
        def coeffs(ext_classes):
            self.setUp()
            ph = self._create_ph_farmer(
                ph_extensions=MultiExtension, max_iter=0,
                extension_kwargs={"ext_classes": ext_classes})
            return ph.extobject.extdict["_CostCoeffRecorder"].coeffs
        cold = coeffs([_CostCoeffRecorder])
        # The reader first, so its pre_iter0 has run when the recorder reads.
        warm = coeffs([wxbarreader.WXBarReader, _CostCoeffRecorder])
        self.assertEqual(cold.keys(), warm.keys())
        for sname in cold:
            for ndn_i, c in cold[sname].items():
                self.assertAlmostEqual(warm[sname][ndn_i], c)


class Test_W_check_multistage(unittest.TestCase):
    """ Dual feasibility of a W file has to hold at every tree node, not just
    summed over all scenarios: hydro has three stages, and W that cancels
    across the stage-2 nodes but not within them is still infeasible."""

    def setUp(self):
        bfs = [3, 3]
        options = {"solver_name": solver_name, "PHIterLimit": 0,
                   "defaultPHrho": 1, "convthresh": 0.001, "verbose": False,
                   "display_progress": False, "display_timing": False}
        self.ph = mpisppy.opt.ph.PH(
            options, [f"Scen{i+1}" for i in range(9)],
            hydro.scenario_creator, hydro.scenario_denouement,
            all_nodenames=sputils.create_nodenames_from_branching_factors(bfs),
            scenario_creator_kwargs={"branching_factors": bfs})
        self.ph.PH_Prep()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _write_W(self, stage2_by_node, root_by_scenario=None):
        root_by_scenario = root_by_scenario or {}
        fname = os.path.join(self.dir.name, "W.csv")
        with open(fname, "w") as f:
            for sname, s in self.ph.local_scenarios.items():
                for node in s._mpisppy_node_list:
                    if node.name == "ROOT":
                        w = root_by_scenario.get(sname, 0.0)
                    else:
                        w = stage2_by_node.get(node.name, 0.0)
                    for v in node.nonant_vardata_list:
                        f.write(f"{sname},{v.name},{w!r}\n")
        return fname

    def test_feasible_W_is_accepted(self):
        wxbarutils.set_W_from_file(self._write_W({}), self.ph, 0)

    def test_large_W_with_rounding_drift_is_accepted(self):
        # E[W] at ROOT is 1e-6 / 9, above an absolute 1e-7, but tiny beside
        # weights of 1e7: the drift PH's own W shows at large rho.
        fname = self._write_W({}, {"Scen1": 1e7 + 1e-6, "Scen2": -1e7})
        wxbarutils.set_W_from_file(fname, self.ph, 0)

    def test_W_infeasible_at_one_node_is_refused(self):
        # +1 under ROOT_0 and -1 under ROOT_1 sum to 0 over all scenarios.
        fname = self._write_W({"ROOT_0": 1.0, "ROOT_1": -1.0})
        with self.assertRaisesRegex(RuntimeError, "at tree node ROOT_0"):
            wxbarutils.set_W_from_file(fname, self.ph, 0)

class _Iter0XbarRecorder(Extension):
    """ Records, in post_iter0, the nonant values and the xbar the rho
    setters see there."""
    def post_iter0(self):
        self.seen = {
            sname: {ndn_i: (v.value, s._mpisppy_model.xbars[ndn_i].value)
                    for ndn_i, v in s._mpisppy_data.nonant_indices.items()}
            for sname, s in self.opt.local_scenarios.items()}


class Test_iter0_xbar_farmer(unittest.TestCase):
    """ What xbar the post_iter0 hooks (SepRho, SensiRho, GradRho) see: the
    mean of the iteration-0 solutions, unless WXBarReader supplied one."""

    def setUp(self):
        self.cfg = _create_cfg()
        self.cfg.sep_rho_args()
        self.cfg.dynamic_rho_args()
        self.cfg.num_scens = 3
        self.cfg.max_iterations = 0

    def _run(self, ext_classes, w_file=False, xbar_file=False):
        cfg = self.cfg
        beans = (cfg, farmer.scenario_creator, farmer.scenario_denouement,
                 farmer.scenario_names_creator(cfg.num_scens))
        hub_dict = vanilla.ph_hub(
            *beans, scenario_creator_kwargs=farmer.kw_creator(cfg),
            ph_extensions=MultiExtension,
            extension_kwargs={"ext_classes": ext_classes})
        options = hub_dict['opt_kwargs']['options']
        options['cfg'] = cfg
        options["sep_rho_options"] = {"multiplier": 1.0, "cfg": cfg}
        if w_file or xbar_file:
            cfg.W_and_xbar_reader = True
            if w_file:
                cfg.init_W_fname = os.path.join(_W_TEST_DATA_DIR, 'w_file.csv')
            if xbar_file:
                cfg.init_Xbar_fname = os.path.join(_W_TEST_DATA_DIR, 'xbar_file.csv')
        wheel = WheelSpinner(hub_dict, [])
        wheel.spin()
        return wheel.spcomm.opt

    def _assert_xbar_is_iter0_mean(self, ph):
        seen = ph.extobject.extdict["_Iter0XbarRecorder"].seen
        mean = {}
        for sname, s in ph.local_scenarios.items():
            for ndn_i, (x, _) in seen[sname].items():
                mean[ndn_i] = mean.get(ndn_i, 0.0) + s._mpisppy_probability * x
        self.assertGreater(max(abs(m) for m in mean.values()), 1.0)
        for sname in seen:
            for ndn_i, (_, xbar) in seen[sname].items():
                self.assertAlmostEqual(xbar, mean[ndn_i], places=6)

    def test_cold_run_sees_iter0_mean(self):
        self._assert_xbar_is_iter0_mean(self._run([_Iter0XbarRecorder]))

    def test_W_only_sees_iter0_mean(self):
        ph = self._run([wxbarreader.WXBarReader, _Iter0XbarRecorder], w_file=True)
        self._assert_xbar_is_iter0_mean(ph)

    def test_supplied_xbar_is_kept(self):
        ph = self._run([wxbarreader.WXBarReader, _Iter0XbarRecorder],
                       w_file=True, xbar_file=True)
        supplied = {}
        with open(os.path.join(_W_TEST_DATA_DIR, 'xbar_file.csv')) as f:
            for vname, val in csv.reader(f):
                supplied[vname] = float(val)
        seen = ph.extobject.extdict["_Iter0XbarRecorder"].seen
        for sname, s in ph.local_scenarios.items():
            for ndn_i, v in s._mpisppy_data.nonant_indices.items():
                self.assertAlmostEqual(seen[sname][ndn_i][1], supplied[v.name])

    def test_sep_rho_uses_iter0_xbar(self):
        # rho = |c| / max(1, E|x - xbar|) for farmer's continuous nonants,
        # with xbar the iteration-0 mean (it used to be 0, giving |c| / E|x|).
        ph = self._run([_Iter0XbarRecorder, SepRho])
        seen = ph.extobject.extdict["_Iter0XbarRecorder"].seen
        sep = ph.extobject.extdict["SepRho"]
        mean = {}
        for sname, s in ph.local_scenarios.items():
            for ndn_i, (x, _) in seen[sname].items():
                mean[ndn_i] = mean.get(ndn_i, 0.0) + s._mpisppy_probability * x
        resid = {}
        for sname, s in ph.local_scenarios.items():
            for ndn_i, (x, _) in seen[sname].items():
                resid[ndn_i] = resid.get(ndn_i, 0.0) \
                    + s._mpisppy_probability * abs(x - mean[ndn_i])
        for s in ph.local_scenarios.values():
            costs = sep.nonant_cost_coeffs(s)
            for ndn_i in s._mpisppy_data.nonant_indices:
                self.assertAlmostEqual(
                    s._mpisppy_model.rho[ndn_i].value,
                    abs(costs[ndn_i]) / max(1, resid[ndn_i]), places=6)


if __name__ == '__main__':
    unittest.main()
