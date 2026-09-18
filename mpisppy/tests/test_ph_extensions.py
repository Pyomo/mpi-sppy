###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Test PH extensions that are otherwise untested:
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
            "iter0_solver_options": {"mipgap": 0.1, "threads": 1},
            "iterk_solver_options": {"mipgap": 0.02, "threads": 1},
            "smoothed": 0,
            "asynchronousPH": False,
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
        # obj can be very large because MultRhoUpdater amplifies rho
        # exponentially, inflating the proximal term; just check it ran
        self.assertIsNotNone(obj)

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
            "iter0_solver_options": {"threads": 1},
            "iterk_solver_options": {"threads": 1},
            "smoothed": 0,
            "asynchronousPH": False,
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
        from mpisppy.utils.w_utils.wtracker import WTracker
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


class TestWtrackerWindow(unittest.TestCase):
    """The window that the three readers of ``local_Ws`` compute.

    No solver and no solve: a WTracker reads W values out of ``local_Ws``,
    so these tests put them there directly. That is what lets the window
    arithmetic be asked about the iteration numbers that expose it -- a set
    of iterations that does not start at 1, a tracker holding nothing --
    rather than about whatever a solve happens to produce.
    """

    WLEN = 3

    def setUp(self):
        options = {
            "solver_name": solver_name,
            "PHIterLimit": 1,
            "defaultPHrho": 1,
            "convthresh": 1e-8,
            "verbose": False,
            "display_timing": False,
            "display_progress": False,
            "smoothed": 0,
            "asynchronousPH": False,
            "toc": False,
        }
        self.ph = mpisppy.opt.ph.PH(
            options,
            [f"Scenario{i+1}" for i in range(3)],
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs={"crops_multiplier": 1},
        )
        self._cleanup_files = []

    def tearDown(self):
        for pattern in self._cleanup_files:
            for f in glob.glob(pattern):
                os.remove(f)

    def _tracker(self, iterations, values=None):
        """A tracker holding one W set at each of ``iterations``.

        ``values`` maps an iteration to the number every W in that set
        takes; iterations it omits get a value that differs from every
        other set's, so a reader that compares the wrong pair still sees a
        difference.
        """
        from mpisppy.utils.w_utils.wtracker import WTracker
        # What grab_local_Ws would have keyed the most recent set by.
        self.ph._PHIter = iterations[-1] if iterations else 0
        tracker = WTracker(self.ph)
        nvars = len(tracker.varnames)
        for iteration in iterations:
            w = (values or {}).get(iteration, float(iteration))
            tracker.local_Ws[iteration] = {
                sname: [w] * nvars
                for sname in self.ph.local_scenario_names}
        self.assertEqual(sorted(tracker.local_Ws), sorted(iterations))
        return tracker

    def test_a_tracker_holding_nothing_reports_instead_of_raising(self):
        """A WTracker built after the run -- the pattern in wtracker's own
        __main__ block -- has grabbed nothing, and asked for a window it
        indexed a dict that has no such keys."""
        tracker = self._tracker([])
        self.ph._PHIter = 5
        tracker.ph_iter = 5
        result = tracker.compute_moving_stats(self.WLEN)
        self.assertIsInstance(
            result, str,
            msg="a tracker holding nothing computed a window instead of "
                "saying it had too few iterations")
        self.assertIn("Not enough iterations tracked", result)

    def test_the_short_window_message_names_the_number_of_sets_needed(self):
        """A window spans fi..li inclusive, so a window of length wlen needs
        wlen+1 sets. Reporting only how many were tracked reads as though
        that number were the threshold, and sizes the next run one short."""
        tracker = self._tracker([5, 6, 7])
        message = tracker.compute_moving_stats(self.WLEN)
        self.assertIn(f"spans {self.WLEN + 1}", message)

    def test_a_window_shorter_than_asked_for_is_reported_not_raised(self):
        """Three sets that do not start at 1, and a window of three.

        One subTest each: they failed differently -- KeyError from the one
        that indexes the window, ValueError from the one that unpacks the
        sentence as though it were a pair of statistics -- and asking them
        in one loop hides the second.
        """
        tracker = self._tracker([5, 6, 7])
        for reader in ("check_cross_zero", "check_w_stdev"):
            with self.subTest(reader=reader):
                answer = (tracker.check_cross_zero(self.WLEN)
                          if reader == "check_cross_zero"
                          else tracker.check_w_stdev(self.WLEN, 1e-3))
                self.assertIsInstance(answer, str)
                self.assertIn("Not enough iterations tracked", answer)
                self.assertTrue(answer.startswith("WTRACKER"), msg=answer)

    def test_the_first_window_of_a_fresh_run_does_not_read_iteration_zero(
            self):
        """Four sets and a window of three is the first window a fresh run
        can report, and iteration 0 is not one of the four."""
        tracker = self._tracker([1, 2, 3, 4])
        answer = tracker.check_cross_zero(self.WLEN)
        self.assertNotIn("Not enough iterations", answer)
        self.assertTrue(answer.startswith("WTRACKER"), msg=answer)

    def test_the_readers_answer_when_the_sets_are_there(self):
        """Sets that start at 5 and a window that fits inside them."""
        tracker = self._tracker([5, 6, 7, 8, 9, 10])
        for reader in ("check_cross_zero", "check_w_stdev"):
            with self.subTest(reader=reader):
                answer = (tracker.check_cross_zero(self.WLEN)
                          if reader == "check_cross_zero"
                          else tracker.check_w_stdev(self.WLEN, 1e-3))
                self.assertNotIn("Not enough iterations", answer)
                self.assertTrue(answer.startswith("WTRACKER"), msg=answer)

    def test_check_cross_zero_compares_each_adjacent_pair(self):
        """The W sets are positive until the last one, so the only crossing
        is at the end of the window. Comparing one fixed pair on every pass
        -- whatever the window -- reports on iteration fi and never sees
        it."""
        tracker = self._tracker([1, 2, 3, 4],
                                values={1: 1.0, 2: 2.0, 3: 3.0, 4: -4.0})
        self.assertEqual(tracker.check_cross_zero(self.WLEN),
                         "WTRACKER BAD: Ws crossed zero, sensed at iter 4")

    def test_check_cross_zero_is_good_when_no_pair_crosses(self):
        tracker = self._tracker([1, 2, 3, 4])
        self.assertIn("GOOD", tracker.check_cross_zero(self.WLEN))

    def test_check_w_stdev_uses_the_offsetback_it_was_given(self):
        """It took offsetback and computed its statistics without it, so it
        answered for the most recent window whatever it was asked about.

        The first four sets are constant -- stdev 0, so GOOD -- and the last
        four swing between 0 and 100. An offsetback of 4 asks about the
        first window; the statistics used to come from the last.
        """
        tracker = self._tracker(
            [1, 2, 3, 4, 5, 6, 7, 8],
            values={1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0,
                    5: 0.0, 6: 100.0, 7: 0.0, 8: 100.0})
        self.assertIn("GOOD", tracker.check_w_stdev(self.WLEN, 0.1,
                                                    offsetback=4))
        # The same tracker, asked about the window it used to answer for.
        self.assertIn("BAD", tracker.check_w_stdev(self.WLEN, 0.1))

    def test_the_report_writes_the_warning_when_the_window_is_short(self):
        """report_by_moving_stats indexed the warning string with [1] and
        wrote its second character as the whole report."""
        prefix = os.path.join(os.path.dirname(__file__), "_test_wtwindow")
        self._cleanup_files.append(prefix + "*")
        tracker = self._tracker([5, 6, 7])
        tracker.report_by_moving_stats(self.WLEN, reportlen=5,
                                       stdevthresh=1e-6, file_prefix=prefix)
        summaries = glob.glob(prefix + "_summary_*")
        self.assertEqual(len(summaries), 1)
        with open(summaries[0]) as f:
            content = f.read()
        self.assertIn("Not enough iterations tracked", content)


if __name__ == '__main__':
    unittest.main()
