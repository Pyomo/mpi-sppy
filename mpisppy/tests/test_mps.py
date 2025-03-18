###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# test mps utilities
import unittest
import mpisppy.utils.mps_reader as mps_reader
from mpisppy.tests.utils import get_solver
import pyomo.environ as pyo
import mip  # pip install mip (from coin-or)

solver_available, solver_name, persistent_available, persistent_solver_name= get_solver(persistent_OK=False)

class TestStochAdmmWrapper(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _reader_body(self, fname):
        pyomo_model = mps_reader.read_mps_and_create_pyomo_model(fname)

        opt = pyo.SolverFactory(solver_name)
        opt.solve(pyomo_model)
        pyomo_obj = pyo.value(pyomo_model.objective)

        m = mip.Model(solver_name="CBC")
        m.read(fname)
        m.optimize()   # returns a status, btw
        coin_obj = m.objective_value
        self.assertAlmostEqual(coin_obj, pyomo_obj, places=3,
                               delta=None, msg=None)
        
    def test_mps_reader_test1(self):
        self._reader_body("examples/test1.mps")
        
    def test_mps_reader_sizes1(self):
        self._reader_body("examples/sizes1.mps")

if __name__ == '__main__':
    unittest.main()
