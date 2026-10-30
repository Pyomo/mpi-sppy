###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Regression tests for bound-vector construction with unbounded variables."""

import types
import unittest

import numpy as np
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver

from mpisppy import MPI
from mpisppy.cylinders.reduced_costs_spoke import ReducedCostsSpoke
from mpisppy.cylinders.reduced_costs_spoke import (
    _lb_generator as _rc_lb_generator,
)
from mpisppy.cylinders.reduced_costs_spoke import (
    _ub_generator as _rc_ub_generator,
)
from mpisppy.cylinders.spcommunicator import SPCommunicator
from mpisppy.cylinders.spwindow import Field
from mpisppy.opt.presolve import _INF as _PRESOLVE_INF
from mpisppy.opt.presolve import (
    _lb_generator as _presolve_lb_generator,
)
from mpisppy.opt.presolve import (
    _ub_generator as _presolve_ub_generator,
)


class TestBoundGenerators(unittest.TestCase):

    def test_unbounded_entries_do_not_shift_later_bounds(self):
        variables = [
            types.SimpleNamespace(lb=None, ub=None),
            types.SimpleNamespace(lb=-2.0, ub=2.0),
            types.SimpleNamespace(lb=None, ub=None),
            types.SimpleNamespace(lb=4.0, ub=8.0),
            types.SimpleNamespace(lb=None, ub=None),
        ]
        for name, lb_generator, ub_generator, infinity in (
            ("reduced costs", _rc_lb_generator, _rc_ub_generator, np.inf),
            ("presolve", _presolve_lb_generator, _presolve_ub_generator,
             _PRESOLVE_INF),
        ):
            expected_lbs = np.array(
                [-infinity, -2.0, -infinity, 4.0, -infinity]
            )
            expected_ubs = np.array(
                [infinity, 2.0, infinity, 8.0, infinity]
            )
            with self.subTest(path=name, bound="lower"):
                actual = np.fromiter(
                    lb_generator(variables), dtype=float, count=len(variables)
                )
                np.testing.assert_array_equal(actual, expected_lbs)
            with self.subTest(path=name, bound="upper"):
                actual = np.fromiter(
                    ub_generator(variables), dtype=float, count=len(variables)
                )
                np.testing.assert_array_equal(actual, expected_ubs)


class _ArrayBuffer:

    def __init__(self, length):
        self.values = np.empty(length)

    def value_array(self):
        return self.values

    def __getitem__(self, key):
        return self.values[key]


class _PersistentPlugin(PersistentSolver):

    def __init__(self):
        self.updated = []

    def update_var(self, var):
        self.updated.append(var)

    def _presolve(self, *args, **kwargs):
        pass

    def _postsolve(self, *args, **kwargs):
        pass

    def _warm_start(self, *args, **kwargs):
        pass

    def _apply_solver(self, *args, **kwargs):
        pass

    def _get_dual_values(self, *args, **kwargs):
        return {}

    def _load_vars(self, *args, **kwargs):
        pass

    def _get_expr_from_pyomo_repn(self, *args, **kwargs):
        pass


class _IntersectionComm:

    def __init__(self, remote_lbs, remote_ubs, remote_position_keys=None):
        self.remote_lbs = np.asarray(remote_lbs)
        self.remote_ubs = np.asarray(remote_ubs)
        self.remote_position_keys = remote_position_keys
        self.calls = []

    def allgather(self, local_position_keys):
        remote_position_keys = self.remote_position_keys
        if remote_position_keys is None:
            remote_position_keys = local_position_keys
        return [local_position_keys, remote_position_keys]

    def Allreduce(self, send, recv, op):
        self.calls.append(op)
        if op == MPI.MAX:
            np.maximum(send, self.remote_lbs, out=recv)
        elif op == MPI.MIN:
            np.minimum(send, self.remote_ubs, out=recv)
        else:
            raise AssertionError(f"unexpected reduction operation {op}")


class TestReducedCostInitialBounds(unittest.TestCase):

    @staticmethod
    def _scenario(bounds):
        variables = [
            types.SimpleNamespace(lb=lb, ub=ub) for lb, ub in bounds
        ]
        return types.SimpleNamespace(
            _mpisppy_data=types.SimpleNamespace(
                nonant_indices=dict(enumerate(variables))
            )
        )

    def test_bounds_are_intersected_across_ranks(self):
        spoke = object.__new__(ReducedCostsSpoke)
        spoke.opt = types.SimpleNamespace(local_scenarios={
            "s0": self._scenario([(0.0, 10.0), (None, 8.0), (-5.0, None)]),
            "s1": self._scenario([(1.0, 9.0), (-2.0, None), (None, 20.0)]),
        })
        spoke.send_buffers = {
            Field.NONANT_LOWER_BOUNDS: _ArrayBuffer(3),
            Field.NONANT_UPPER_BOUNDS: _ArrayBuffer(3),
        }
        spoke.nonant_length = 3
        spoke.cylinder_comm = _IntersectionComm(
            remote_lbs=[3.0, -1.0, -7.0],
            remote_ubs=[7.0, 6.0, 15.0],
        )

        spoke.initialize_bound_fields()

        np.testing.assert_array_equal(
            spoke._nonant_lower_bounds, [3.0, -1.0, -5.0]
        )
        np.testing.assert_array_equal(
            spoke._nonant_upper_bounds, [7.0, 6.0, 15.0]
        )
        self.assertEqual(spoke.cylinder_comm.calls, [MPI.MAX, MPI.MIN])

    def test_unrelated_multistage_nodes_are_not_intersected(self):
        spoke = object.__new__(ReducedCostsSpoke)
        spoke.opt = types.SimpleNamespace(local_scenarios={
            "branch_a": types.SimpleNamespace(
                _mpisppy_data=types.SimpleNamespace(nonant_indices={
                    ("ROOT", 0): types.SimpleNamespace(lb=0.0, ub=10.0),
                    ("ROOT_0", 0): types.SimpleNamespace(lb=0.0, ub=1.0),
                })
            )
        })
        spoke.send_buffers = {
            Field.NONANT_LOWER_BOUNDS: _ArrayBuffer(2),
            Field.NONANT_UPPER_BOUNDS: _ArrayBuffer(2),
        }
        spoke.nonant_length = 2
        spoke.cylinder_comm = _IntersectionComm(
            remote_lbs=[-5.0, 10.0],
            remote_ubs=[20.0, 20.0],
            remote_position_keys=[("ROOT", 0), ("ROOT_1", 0)],
        )

        spoke.initialize_bound_fields()

        # ROOT is common and is intersected. The second position names
        # independent branch decisions, so tightening there is disabled rather
        # than producing the invalid positional intersection [10, 1].
        np.testing.assert_array_equal(
            spoke._nonant_lower_bounds, [0.0, -np.inf]
        )
        np.testing.assert_array_equal(
            spoke._nonant_upper_bounds, [10.0, np.inf]
        )

    def test_bundle_bounds_include_every_consensus_variable(self):
        indices = [("ROOT", 0), ("ROOT", 1)]
        ref_vars = {
            indices[0]: types.SimpleNamespace(lb=0.0, ub=10.0),
            indices[1]: types.SimpleNamespace(lb=None, ub=None),
        }
        tied_vars = {
            indices[0]: types.SimpleNamespace(lb=2.0, ub=7.0),
            indices[1]: types.SimpleNamespace(lb=-3.0, ub=4.0),
        }
        bundle = types.SimpleNamespace(
            _ef_scenario_names=["s0", "s1"],
            _mpisppy_data=types.SimpleNamespace(nonant_indices=ref_vars),
            consensus_groups={
                ndn_i: [ref_vars[ndn_i], tied_vars[ndn_i]]
                for ndn_i in indices
            },
        )

        spoke = object.__new__(ReducedCostsSpoke)
        spoke.opt = types.SimpleNamespace(local_scenarios={"bundle": bundle})
        spoke.send_buffers = {
            Field.NONANT_LOWER_BOUNDS: _ArrayBuffer(2),
            Field.NONANT_UPPER_BOUNDS: _ArrayBuffer(2),
        }
        spoke.nonant_length = 2
        spoke.cylinder_comm = _IntersectionComm(
            remote_lbs=[-np.inf, -np.inf],
            remote_ubs=[np.inf, np.inf],
            remote_position_keys=indices,
        )

        spoke.initialize_bound_fields()

        np.testing.assert_array_equal(
            spoke._nonant_lower_bounds, [2.0, -3.0]
        )
        np.testing.assert_array_equal(
            spoke._nonant_upper_bounds, [7.0, 4.0]
        )


class TestPersistentBoundUpdates(unittest.TestCase):

    @staticmethod
    def _scenario(plugin):
        variables = [
            types.SimpleNamespace(lb=0.0, ub=10.0),
            types.SimpleNamespace(lb=0.0, ub=10.0),
        ]
        return (
            types.SimpleNamespace(
                _solver_plugin=plugin,
                _mpisppy_data=types.SimpleNamespace(
                    nonant_indices=dict(enumerate(variables))
                ),
            ),
            variables,
        )

    def test_reduced_cost_spoke_updates_each_modified_var_once(self):
        plugin = _PersistentPlugin()
        scenario, variables = self._scenario(plugin)
        spoke = object.__new__(ReducedCostsSpoke)
        spoke.opt = types.SimpleNamespace(local_scenarios={"s0": scenario})
        spoke.send_buffers = {
            Field.NONANT_LOWER_BOUNDS: _ArrayBuffer(2),
            Field.NONANT_UPPER_BOUNDS: _ArrayBuffer(2),
        }
        spoke.send_buffers[Field.NONANT_LOWER_BOUNDS].values[:] = [1.0, 0.0]
        spoke.send_buffers[Field.NONANT_UPPER_BOUNDS].values[:] = [9.0, 10.0]
        spoke.cylinder_rank = 1

        spoke.update_nonant_bounds()

        self.assertEqual((variables[0].lb, variables[0].ub), (1.0, 9.0))
        self.assertEqual((variables[1].lb, variables[1].ub), (0.0, 10.0))
        self.assertEqual(plugin.updated, [variables[0]])

    def test_receiver_updates_each_modified_var_once(self):
        plugin = _PersistentPlugin()
        scenario, variables = self._scenario(plugin)
        lower = _ArrayBuffer(2)
        upper = _ArrayBuffer(2)
        lower.values[:] = [1.0, 0.0]
        upper.values[:] = [9.0, 10.0]
        communicator = types.SimpleNamespace(
            receive_field_spcomms={
                Field.NONANT_LOWER_BOUNDS: [(1, None, lower)],
                Field.NONANT_UPPER_BOUNDS: [(1, None, upper)],
            },
            opt=types.SimpleNamespace(local_scenarios={"s0": scenario}),
            cylinder_rank=1,
            get_receive_buffer=lambda *args: True,
        )

        SPCommunicator.receive_nonant_bounds(communicator)

        self.assertEqual((variables[0].lb, variables[0].ub), (1.0, 9.0))
        self.assertEqual((variables[1].lb, variables[1].ub), (0.0, 10.0))
        self.assertEqual(plugin.updated, [variables[0]])


if __name__ == "__main__":
    unittest.main()
