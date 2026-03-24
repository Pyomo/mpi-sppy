###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Test PH extensions that are otherwise untested:
#   - Gapper (mipgapper.py, 0% coverage)
#   - MultRhoUpdater (mult_rho_updater.py, 24% coverage)
#   - Wtracker_extension (wtracker_extension.py, 0%) + WTracker (wtracker.py, 27%)

import glob
import os
import unittest

import mpisppy.opt.ph
import mpisppy.tests.examples.farmer as farmer
from mpisppy.tests.examples.sizes.sizes import scenario_creator as sizes_creator, \
                                               scenario_denouement as sizes_denouement
from mpisppy.tests.utils import get_solver
import mpisppy.MPI as mpi

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()


class TestGapper(unittest.TestCase):
    """Test the Gapper (mipgapper) extension with sizes."""

    def setUp(self):
        self.options = {
            "solver_name": solver_name,
            "PHIterLimit": 5,
            "defaultPHrho": 1,
            "convthresh": 0.001,
            "verbose": False,
            "display_timing": False,
            "display_progress": False,
            "iter0_solver_options": {"mipgap": 0.1},
            "iterk_solver_options": {"mipgap": 0.02},
            "smoothed": 0,
            "asynchronousPH": False,
            "subsolvedirectives": None,
            "toc": False,
        }
        self.scenario_names = [f"Scenario{i+1}" for i in range(3)]
        self.creator_kwargs = {"scenario_count": 3}

    def _copy_options(self):
        return dict(self.options)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_gapper_dict_mode(self):
        """Gapper with mipgapdict should set mipgap per iteration."""
        from mpisppy.extensions.mipgapper import Gapper
        options = self._copy_options()
        options["gapperoptions"] = {
            "mipgapdict": {0: 0.10, 2: 0.05, 4: 0.01},
        }
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            sizes_creator,
            sizes_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            extensions=Gapper,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)
        # After iteration 4, mipgap should have been set to 0.01
        self.assertAlmostEqual(
            ph.current_solver_options["mipgap"], 0.01, places=5)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_gapper_changes_gap(self):
        """Verify Gapper actually changes the gap across iterations."""
        from mpisppy.extensions.mipgapper import Gapper
        options = self._copy_options()
        options["PHIterLimit"] = 3
        options["gapperoptions"] = {
            "mipgapdict": {0: 0.50, 2: 0.001},
        }
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            sizes_creator,
            sizes_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            extensions=Gapper,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)
        # Verify the final gap was applied
        self.assertAlmostEqual(
            ph.current_solver_options["mipgap"], 0.001, places=5)


class TestMultRhoUpdater(unittest.TestCase):
    """Test the MultRhoUpdater extension."""

    def setUp(self):
        self.options = {
            "solver_name": solver_name,
            "PHIterLimit": 10,
            "defaultPHrho": 1,
            "convthresh": 0.001,
            "verbose": False,
            "display_timing": False,
            "display_progress": False,
            "iter0_solver_options": {"mipgap": 0.1},
            "iterk_solver_options": {"mipgap": 0.02},
            "smoothed": 0,
            "asynchronousPH": False,
            "subsolvedirectives": None,
            "toc": False,
        }
        self.scenario_names = [f"Scenario{i+1}" for i in range(3)]
        self.creator_kwargs = {"scenario_count": 3}

    def _copy_options(self):
        return dict(self.options)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_mult_rho_updater_runs(self):
        """MultRhoUpdater should adjust rho values during PH."""
        from mpisppy.extensions.mult_rho_updater import MultRhoUpdater
        options = self._copy_options()
        options["mult_rho_options"] = {
            "convergence_tolerance": 1e-4,
            "rho_update_stop_iteration": None,
            "rho_update_start_iteration": 2,
            "verbose": False,
        }
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            sizes_creator,
            sizes_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            extensions=MultRhoUpdater,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)
        self.assertGreater(obj, 100000)
        self.assertLess(obj, 400000)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_mult_rho_updater_default_options(self):
        """MultRhoUpdater should work with no explicit mult_rho_options."""
        from mpisppy.extensions.mult_rho_updater import MultRhoUpdater
        options = self._copy_options()
        options["PHIterLimit"] = 5
        # Don't set mult_rho_options -- it should use defaults
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            sizes_creator,
            sizes_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            extensions=MultRhoUpdater,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)


class TestWtrackerExtension(unittest.TestCase):
    """Test Wtracker_extension and the underlying WTracker utility."""

    def setUp(self):
        self.options = {
            "solver_name": solver_name,
            "PHIterLimit": 8,
            "defaultPHrho": 1,
            "convthresh": 1e-8,
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
        self._cleanup_files = []

    def tearDown(self):
        for pattern in self._cleanup_files:
            for f in glob.glob(pattern):
                os.remove(f)

    def _copy_options(self):
        return dict(self.options)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_wtracker_extension(self):
        """Wtracker_extension should track W values and produce report files."""
        from mpisppy.extensions.wtracker_extension import Wtracker_extension
        prefix = os.path.join(os.path.dirname(__file__), "_test_wt")
        self._cleanup_files.append(prefix + "*")
        options = self._copy_options()
        options["wtracker_options"] = {
            "wlen": 3,
            "reportlen": 5,
            "stdevthresh": 1e-6,
            "file_prefix": prefix,
        }
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            extensions=Wtracker_extension,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)
        # Check that summary report was created
        summary_files = glob.glob(prefix + "_summary_*")
        self.assertGreater(len(summary_files), 0,
                           "Wtracker should produce summary files")
        # Check summary file has content
        with open(summary_files[0]) as f:
            content = f.read()
        self.assertIn("nonants", content)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_wtracker_direct(self):
        """WTracker used directly should compute moving stats."""
        from mpisppy.utils.wtracker import WTracker
        options = self._copy_options()
        options["PHIterLimit"] = 8
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj, tbound = ph.ph_main()
        # Now use WTracker directly on the post-PH object
        wt = WTracker(ph)
        # Simulate grabbing Ws over several "iterations"
        for i in range(6):
            ph._PHIter = i
            wt.grab_local_Ws()
        # With 6 iterations and wlen=3, we should get real stats
        result = wt.compute_moving_stats(wlen=3)
        # Should return (wlist, window_stats) tuple, not a warning string
        self.assertIsInstance(result, tuple)
        wlist, window_stats = result
        self.assertGreater(len(window_stats), 0)
        # Each stat entry should be (mean, stdev)
        for key, (mean, stdev) in window_stats.items():
            self.assertIsNotNone(mean)
            self.assertGreaterEqual(stdev, 0)


if __name__ == '__main__':
    unittest.main()
