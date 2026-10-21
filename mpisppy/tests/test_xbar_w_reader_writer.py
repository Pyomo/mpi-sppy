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

import mpisppy.tests.examples.farmer as farmer
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.w_utils.wxbarreader as wxbarreader
import mpisppy.utils.w_utils.wxbarwriter as wxbarwriter
from mpisppy.extensions.extension import Extension, MultiExtension
from mpisppy.extensions.sep_rho import SepRho
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver
from mpisppy.utils import config

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

#*****************************************************************************

class Test_xbar_w_reader_writer_farmer(unittest.TestCase):
    """ Test the gradient code using farmer."""

    def _create_ph_farmer(self, ph_extensions=None, max_iter=100):
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
        hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs, ph_extensions=ph_extensions)
        hub_dict['opt_kwargs']['options']['cfg'] = self.cfg
        if ph_extensions==wxbarwriter.WXBarWriter:
            self.cfg.W_and_xbar_writer = True
            self.cfg.W_fname = self.temp_w_file_name
            self.cfg.Xbar_fname = self.temp_xbar_file_name
        if ph_extensions==wxbarreader.WXBarReader:
            self.cfg.W_and_xbar_reader = True
            self.cfg.init_W_fname = self.w_file_name
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
