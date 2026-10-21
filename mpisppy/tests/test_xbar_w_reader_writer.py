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
import unittest

import pyomo.environ as pyo

import mpisppy.tests.examples.farmer as farmer
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.w_utils.wxbarreader as wxbarreader
import mpisppy.utils.w_utils.wxbarwriter as wxbarwriter
from mpisppy.extensions.extension import Extension, MultiExtension
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


if __name__ == '__main__':
    unittest.main()
