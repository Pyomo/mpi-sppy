###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Test PH.ph_main() directly (not through the cylinder hub system).
# This exercises the standalone PH code path, including extensions
# and convergers that are otherwise untested.

import os
import shutil
import unittest

import mpisppy.opt.ph
import mpisppy.tests.examples.farmer as farmer
from mpisppy.tests.utils import get_solver
import mpisppy.MPI as mpi

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()


class TestPHMainFarmer(unittest.TestCase):
    """Test PH.ph_main() with the farmer model."""

    def setUp(self):
        self.options = {
            "solver_name": solver_name,
            "PHIterLimit": 10,
            "defaultPHrho": 1,
            "convthresh": 0.001,
            "verbose": False,
            "display_timing": False,
            "display_progress": False,
            "iter0_solver_options": None,
            "iterk_solver_options": None,
            "smoothed": 0,
            "asynchronousPH": False,
            "subsolvedirectives": None,
            "toc": False,
        }
        self.scenario_names = [f"Scenario{i+1}" for i in range(3)]
        self.creator_kwargs = {"crops_multiplier": 1}

    def _copy_options(self):
        return dict(self.options)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_ph_main_basic(self):
        """Basic PH.ph_main() smoke test with farmer."""
        ph = mpisppy.opt.ph.PH(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)
        self.assertIsNotNone(tbound)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_ph_main_no_finalize(self):
        """PH.ph_main() with finalize=False."""
        ph = mpisppy.opt.ph.PH(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj, tbound = ph.ph_main(finalize=False)
        self.assertIsNone(obj)
        self.assertIsNotNone(tbound)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_ph_main_with_xhatlooper(self):
        """PH with XhatLooper extension."""
        from mpisppy.extensions.xhatlooper import XhatLooper
        options = self._copy_options()
        options["xhat_looper_options"] = {
            "xhat_solver_options": None,
            "scen_limit": 3,
        }
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            extensions=XhatLooper,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_ph_main_with_frac_converger(self):
        """PH with FractionalConverger (farmer has no integers so converges immediately)."""
        from mpisppy.convergers.fracintsnotconv import FractionalConverger
        ph = mpisppy.opt.ph.PH(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            ph_converger=FractionalConverger,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_ph_main_with_diagnoser(self):
        """PH with Diagnoser extension."""
        from mpisppy.extensions.diagnoser import Diagnoser
        diagdir = os.path.join(os.path.dirname(__file__), "_test_diagdir")
        if os.path.exists(diagdir):
            shutil.rmtree(diagdir)
        options = self._copy_options()
        options["PHIterLimit"] = 3
        options["diagnoser_options"] = {"diagnoser_outdir": diagdir}
        try:
            ph = mpisppy.opt.ph.PH(
                options,
                self.scenario_names,
                farmer.scenario_creator,
                farmer.scenario_denouement,
                scenario_creator_kwargs=self.creator_kwargs,
                extensions=Diagnoser,
            )
            conv, obj, tbound = ph.ph_main()
            self.assertIsNotNone(obj)
        finally:
            if os.path.exists(diagdir):
                shutil.rmtree(diagdir)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_ph_main_with_avgminmaxer(self):
        """PH with MinMaxAvg extension."""
        from mpisppy.extensions.avgminmaxer import MinMaxAvg
        options = self._copy_options()
        options["PHIterLimit"] = 3
        options["avgminmax_name"] = "FirstStageCost"
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            extensions=MinMaxAvg,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)


if __name__ == '__main__':
    unittest.main()
