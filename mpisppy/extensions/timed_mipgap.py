###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2025, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
'''
This class is implemented as an extension to be used in mpi-sppy to add a termination callback to a
persistent solver and implement a time-dependent target MIP gap.
'''

import math

import mpisppy.extensions.extension
import mpisppy.utils.sputils as sputils
from mpisppy.utils.callbacks.termination import (
    set_termination_callback,
    supports_termination_callback,
)

class TimedMIPGapCB(mpisppy.extensions.extension.Extension):
    '''
    This extension adds a solver termination callback that implements a time-dependent target MIP gap.
    The curve is defined by an ordered mapping of gap:time pairs, monotonically increasing in both
    dimensions. For each (t,g) pair, when the solver reaches time t, it terminates if the relative gap
    has already improved below g.

    This class requires the following options:
    'timed_mipgap':
        'timecurve': string of gap:time pairs separated by spaces. Example: "0.02:100 0.05:200"

    Attributes
    ----------
    timecurve: ordered dict of {gap:time} pairs.
    '''

    def __init__(self, ph):
        super().__init__(ph)
        self.ph = ph
        self._set_options()

    def _set_options(self):
        ph = self.ph
        if 'timed_mipgap' not in ph.options:
            raise RuntimeError('Did not find "timed_mipgap" options')
        timecurve_str = ph.options['timed_mipgap']['timecurve']
        self.timecurve = dict()
        prev_gap = None
        prev_time = None
        for entry in timecurve_str.split():
            try:
                gap_str, time_str = entry.split(':', 1)
                gap = float(gap_str)
                solve_time = float(time_str)
            except ValueError as exc:
                raise RuntimeError(
                    'Timed MIP gap option "timecurve" entries must have format "gap:time"'
                ) from exc

            if not math.isfinite(gap) or not math.isfinite(solve_time):
                raise RuntimeError(
                    'Timed MIP gap option "timecurve" entries must use finite gap and time values'
                )
            if gap in self.timecurve:
                raise RuntimeError(
                    f'Timed MIP gap option "timecurve" has duplicate gap entry {gap}'
                )
            if prev_gap is not None and (gap <= prev_gap or solve_time <= prev_time):
                raise RuntimeError(
                    'Timed MIP gap option "timecurve" must be strictly increasing in both gap and time'
                )

            self.timecurve[gap] = solve_time
            prev_gap = gap
            prev_time = solve_time

        if not self.timecurve:
            raise RuntimeError('Timed MIP gap option "timecurve" must not be empty')

    @staticmethod
    def _compute_relative_gap(best_obj, best_bound):
        '''Return relative gap or None if a bound is unavailable.'''
        if best_obj is None or best_bound is None:
            return None
        if not math.isfinite(best_obj) or not math.isfinite(best_bound):
            return None

        return abs(best_obj - best_bound) / max(
            1e-10,
            abs(best_obj),
            abs(best_bound),
        )

    def _should_terminate(self, runtime, best_obj, best_bound):
        '''Return True when the timecurve says to terminate.'''
        if runtime is None or not math.isfinite(runtime):
            return False

        gap = self._compute_relative_gap(best_obj, best_bound)
        if gap is None:
            return False

        for tc_gap, tc_t in self.timecurve.items():
            if runtime > tc_t and gap < tc_gap:
                return True

        return False

    def pre_solve(self, s):
        if not hasattr(s, '_solver_plugin'):
            raise RuntimeError('Solver must be created before calling callback extension')
        if not (sputils.is_persistent(s._solver_plugin)):
            raise RuntimeError('Solvers must be persistent for callback definition')
        if not s._solver_plugin.has_instance():
            raise RuntimeError('Solver must be instantiated before calling callback extension')
        if not supports_termination_callback(s._solver_plugin):
            raise RuntimeError(
                'Timed MIP gap requires a persistent solver with supported termination callbacks'
            )

        def cb_fun(runtime, best_obj, best_bound):
            return self._should_terminate(runtime, best_obj, best_bound)

        set_termination_callback(s._solver_plugin, cb_fun)
