# Copyright 2023 by U. Naepels and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Author:
"""
IMPORTANT:
  Unless we run to convergence, the solver, and even solver
version matter a lot, so we often just do smoke tests.
"""

import os
import glob
import unittest
import pandas as pd
import csv
import pyomo.environ as pyo
import mpisppy.opt.ph
import mpisppy.phbase
from mpisppy.utils import config

import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.sputils as sputils
import mpisppy.utils.rho_utils as rho_utils
import mpisppy.confidence_intervals.ciutils as ciutils
import mpisppy.tests.examples.farmer as farmer
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver,round_pos_sig
import mpisppy.utils.gradient as grad
import mpisppy.utils.find_rho as find_rho
from mpisppy.utils.wxbarwriter import WXBarWriter
from mpisppy.utils.wxbarreader import WXBarReader


__version__ = 0.1

solver_available,solver_name, persistent_available, persistent_solver_name= get_solver()

def _create_cfg():
    cfg = config.Config()
    cfg.add_branching_factors()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.solver_name = solver_name
    cfg.default_rho = 1
    return cfg

#*****************************************************************************

class Test_w_writer_farmer(unittest.TestCase):
    """ Test the gradient code using farmer."""

    def _create_ph_farmer(self, ph_extensions=None, max_iter=100):
        self.w_file_name = './examples/w_test_data/w_file.csv'
        self.temp_w_file_name = './examples/w_test_data/_temp_w_file.csv'
        self.xbar_file_name = './examples/w_test_data/xbar_file.csv'
        self.temp_xbar_file_name = './examples/w_test_data/_temp_xbar_file.csv'
        self.cfg.num_scens = 3
        scenario_creator = farmer.scenario_creator
        scenario_denouement = farmer.scenario_denouement
        all_scenario_names = farmer.scenario_names_creator(self.cfg.num_scens)
        scenario_creator_kwargs = farmer.kw_creator(self.cfg)
        self.cfg.max_iterations = max_iter
        beans = (self.cfg, scenario_creator, scenario_denouement, all_scenario_names)
        hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs, ph_extensions=ph_extensions)
        if ph_extensions==WXBarWriter: #tbd
            hub_dict['opt_kwargs']['options']["W_and_xbar_writer"] =  {"Wcsvdir": "Wdir"}
            hub_dict['opt_kwargs']['options']['W_fname'] = self.temp_w_file_name
            hub_dict['opt_kwargs']['options']['Xbar_fname'] = self.temp_xbar_file_name
        if ph_extensions==WXBarReader:
            hub_dict['opt_kwargs']['options']["W_and_xbar_reader"] =  {"Wcsvdir": "Wdir"}
            hub_dict['opt_kwargs']['options']['init_W_fname'] = self.w_file_name
            hub_dict['opt_kwargs']['options']['init_Xbar_fname'] = self.xbar_file_name
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
        self.ph_object = self._create_ph_farmer(ph_extensions=WXBarWriter, max_iter=5)
        with open(self.temp_w_file_name, 'r') as f:
            read = csv.reader(f)
            rows = list(read)
            self.assertAlmostEqual(float(rows[1][2]), 70.84705093609978)
            self.assertAlmostEqual(float(rows[3][2]), -41.104251445950844)
        os.remove(self.temp_w_file_name)

    def test_xbarwriter(self):
        self.ph_object = self._create_ph_farmer(ph_extensions=WXBarWriter, max_iter=5)
        with open(self.temp_xbar_file_name, 'r') as f:
            read = csv.reader(f)
            rows = list(read)
            self.assertAlmostEqual(float(rows[1][1]), 274.2239371483933)
            self.assertAlmostEqual(float(rows[3][1]), 96.88717449844287)
        os.remove(self.temp_xbar_file_name)

    def test_wreader(self):
        self.ph_object = self._create_ph_farmer(ph_extensions=WXBarReader, max_iter=0)
        for sname, scenario in self.ph_object.local_scenarios.items():
            if sname == 'scen0':
                self.assertAlmostEqual(scenario._mpisppy_model.W[("ROOT", 1)]._value, 70.84705093609978)
            if sname == 'scen1':
                self.assertAlmostEqual(scenario._mpisppy_model.W[("ROOT", 0)]._value, -41.104251445950844)

    def test_xbarreader(self):
        self.ph_object = self._create_ph_farmer(ph_extensions=WXBarReader, max_iter=0)
        for sname, scenario in self.ph_object.local_scenarios.items():
            if sname == 'scen0':
                self.assertAlmostEqual(scenario._mpisppy_model.xbars[("ROOT", 1)]._value, 274.2239371483933)
            if sname == 'scen1':
                self.assertAlmostEqual(scenario._mpisppy_model.xbars[("ROOT", 0)]._value, 96.88717449844287)




if __name__ == '__main__':
    unittest.main()
