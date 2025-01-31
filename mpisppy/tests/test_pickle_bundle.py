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
import tempfile
import unittest

import mpisppy.MPI as mpi

from mpisppy.tests.utils import get_solver


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
        cmdstr = f"python bundle_pickler.py {self.BF_str} --pickle-bundles-dir={self.tempdir_name} --scenarios-per-bundle={self.SPB} {self.EC}"
        ret = os.system(cmdstr)
        if ret != 0:
            raise RuntimeError(f"Test run failed with code {ret}")

    def test_chain(self):
        # run the pickle bundler then aircond_cylinders
        cmdstr = f"python bundle_pickler.py {self.BF_str} --pickle-bundles-dir={self.tempdir_name} --scenarios-per-bundle={self.SPB} {self.EC}"
        ret = os.system(cmdstr)
        if ret != 0:
            raise RuntimeError(f"pickler part of test run failed with code {ret}")
        
        cmdstr = f"python aircond_cylinders.py --branching-factors=\"{self.BPF}\" --unpickle-bundles-dir={self.tempdir_name} --scenarios-per-bundle={self.SPB} {self.EC} "+\
                 f"--default-rho=1 --max-solver-threads=2 --bundles-per-rank=0 --max-iterations=2 --solver-name={solver_name}"
        ret = os.system(cmdstr)
        if ret != 0:
            raise RuntimeError(f"cylinders part of test run failed with code {ret}")

if __name__ == '__main__':
    unittest.main()
    
