# This software is distributed under the 3-clause BSD License.
# Provide some test for aph under mpisppy.0
# Author: David L. Woodruff (circa September 2019)
"""
IMPORTANT:
  Unless we run to convergence and check xhat, the solver, and even solver
version matter a lot, so we often just do smoke tests.
"""

import pyutilib.th as unittest
from math import log10, floor
import pyomo.environ as pyo
import mpisppy.opt.aph
import mpisppy.phbase
from mpisppy.examples.sizes.sizes import scenario_creator, \
                                       scenario_denouement, \
                                       _rho_setter

__version__ = 0.31

import mpi4py.MPI as mpi
fullcomm = mpi.COMM_WORLD
rank_global = fullcomm.Get_rank()

solvers = ["xpress", "gurobi", "cplex"]

for solvername in solvers:
    solver_available = pyo.SolverFactory(solvername).available()
    if solver_available:
        break

persistentsolvername = solvername+"_persistent"
persistent_available = pyo.SolverFactory(persistentsolvername).available()

#*****************************************************************************
class Test_aph_sizes(unittest.TestCase):
    """ Test the aph mpisppy code using sizes."""

    def setUp(self):
        self.BasePHoptions = {}
        self.BasePHoptions["asynchronousPH"] = False
        self.BasePHoptions["solvername"] = solvername
        self.BasePHoptions["PHIterLimit"] = 10
        self.BasePHoptions["defaultPHrho"] = 1
        self.BasePHoptions["convthresh"] = 0.001
        self.BasePHoptions["subsolvedirectives"] = None
        self.BasePHoptions["verbose"] = False
        self.BasePHoptions["display_timing"] = False
        self.BasePHoptions["display_progress"] = False
        self.BasePHoptions["iter0_solver_options"] = {"mipgap": 0.001}
        self.BasePHoptions["iterk_solver_options"] = {"mipgap": 0.00001}

        self.BasePHoptions["display_progress"] = False

        self.all3_scenario_names = list()
        for sn in range(3):
            self.all3_scenario_names.append("Scenario"+str(sn+1))

        self.all10_scenario_names = list()
        for sn in range(10):
            self.all10_scenario_names.append("Scenario"+str(sn+1))

    def _copy_of_base_options(self):
        retval = {}
        for k,v in self.BasePHoptions.items():
            retval[k] = v
        return retval

    def round_pos_sig(self, x, sig=1):
        return round(x, sig-int(floor(log10(abs(x))))-1)
        
    def test_make_aph(self):
        """ Just smoke to verify construction"""
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        PHoptions["async_frac_needed"] = 0.5
        PHoptions["async_sleep_secs"] = 0.5
        aph = mpisppy.opt.aph.APH(PHoptions,
                              self.all3_scenario_names,
                              scenario_creator,
                              scenario_denouement,
                              cb_data=3)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solvername,))
    def test_aph_basic(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        PHoptions["async_frac_needed"] = 0.5
        PHoptions["async_sleep_secs"] = 0.5
        aph = mpisppy.opt.aph.APH(PHoptions,
                              self.all3_scenario_names,
                              scenario_creator,
                              scenario_denouement,
                              cb_data=3)

        conv, obj, tbound = aph.APH_main(spcomm=None)
        print ("objthing={}, (was=-2435908)".format(obj))
        print ("tbound ={} (was=224106)".format(tbound))

    def test_bundles(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        PHoptions["bundles_per_rank"] = 2
        PHoptions["async_frac_needed"] = 0.5
        PHoptions["async_sleep_secs"] = 0.5
        aph = mpisppy.opt.aph.APH(PHoptions, self.all10_scenario_names,
                              scenario_creator, scenario_denouement,
                              cb_data=10)
        conv, obj, tbound = aph.APH_main(spcomm=None)
        print ("objthing={}, (was=224712.9)".format(obj))
        print ("tbound ={} (was=223168.5)".format(tbound))


if __name__ == '__main__':
    unittest.main()
