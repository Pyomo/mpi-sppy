###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""APH-aware W-spoke acquisition and dual-feasibility projection tests."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pyomo.environ as pyo
import pytest

from mpisppy import MPI
from mpisppy.cylinders.hub import APHHub, PHHub
from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
from mpisppy.cylinders.reduced_costs_spoke import ReducedCostsSpoke
from mpisppy.cylinders.spcommunicator import ReadConsistency
from mpisppy.cylinders.spoke import _BoundNonantLenSpoke, _BoundWSpoke
from mpisppy.cylinders.spwindow import Field
from mpisppy.phbase import Wbar_by_node, project_W_to_zero_sum


def _bare_lagrangian_spoke(asynchronous):
    spoke = object.__new__(LagrangianOuterBound)
    spoke._dual_source_is_asynchronous = asynchronous
    spoke._nonant_len_receive_buffer = mock.Mock()
    spoke._nonant_len_receive_rank = 0
    return spoke


def test_reduced_costs_inherits_aph_dual_handling():
    assert issubclass(ReducedCostsSpoke, LagrangianOuterBound)


@pytest.mark.parametrize(
    ("source_class", "expected"),
    [(APHHub, True), (PHHub, False)],
)
def test_w_spoke_detects_source_publication_policy(source_class, expected):
    spoke = _bare_lagrangian_spoke(asynchronous=False)
    receive_values = np.full(3, np.nan)
    spoke._nonant_len_receive_buffer.value_array.return_value = receive_values
    spoke.communicators = [{"spcomm_class": source_class}]
    with mock.patch.object(
            _BoundNonantLenSpoke, "register_receive_fields"):
        _BoundWSpoke.register_receive_fields(spoke)

    assert spoke._dual_source_is_asynchronous is expected
    if expected:
        np.testing.assert_array_equal(receive_values, np.zeros(3))
    else:
        assert np.all(np.isnan(receive_values))


def test_synchronous_dual_source_keeps_strict_read():
    spoke = _bare_lagrangian_spoke(asynchronous=False)
    spoke._update_nonant_len_buffer = mock.Mock(return_value=True)
    spoke.get_receive_buffer = mock.Mock(
        side_effect=AssertionError("asynchronous read should not be used"))

    assert spoke.update_Ws() is True
    spoke._update_nonant_len_buffer.assert_called_once_with()


def test_asynchronous_dual_source_reads_latest_and_agrees_to_solve():
    spoke = _bare_lagrangian_spoke(asynchronous=True)
    spoke.get_receive_buffer = mock.Mock(return_value=True)
    spoke.allreduce_or = mock.Mock(return_value=True)

    assert spoke.update_Ws() is True
    spoke.get_receive_buffer.assert_called_once_with(
        spoke._nonant_len_receive_buffer,
        Field.DUALS,
        0,
        consistency=ReadConsistency.ASYNC_LATEST,
    )
    spoke.allreduce_or.assert_called_once_with(True)


def test_all_spoke_ranks_solve_when_only_one_rank_received_an_update():
    spoke = _bare_lagrangian_spoke(asynchronous=True)
    receive_values = np.array([1.0, 2.0])
    spoke._nonant_len_receive_buffer.value_array.return_value = receive_values

    def stale_poll(*args, **kwargs):
        # Equal-rank get_receive_buffer reads before it discovers the source ID
        # did not advance. The W-spoke must undo that speculative overwrite.
        receive_values[:] = np.nan
        return False

    spoke.get_receive_buffer = mock.Mock(side_effect=stale_poll)
    # The real allreduce can return true here because a peer spoke rank read a
    # newer APH rank even though this rank retained its previous local piece.
    spoke.allreduce_or = mock.Mock(return_value=True)

    assert spoke.update_Ws() is True
    spoke.allreduce_or.assert_called_once_with(False)
    np.testing.assert_array_equal(receive_values, [1.0, 2.0])


def test_lagrangian_projects_async_weights_before_solving():
    events = []
    spoke = _bare_lagrangian_spoke(asynchronous=True)
    weights = object()
    spoke._nonant_len_receive_buffer.value_array.return_value = weights
    spoke.opt = SimpleNamespace(
        W_from_flat_list=lambda values: events.append(("load", values)))
    spoke.lagrangian = lambda warmstart=False: events.append("solve") or 12.0

    with mock.patch(
            "mpisppy.cylinders.lagrangian_bounder.ph_base.project_W_to_zero_sum",
            side_effect=lambda opt: events.append(("project", opt))):
        assert spoke._set_weights_and_solve() == 12.0

    assert events == [
        ("load", weights),
        ("project", spoke.opt),
        "solve",
    ]


class _Scenario:
    """Hashable scenario holder for the projection test."""

    def __init__(self, probability, weights):
        model = pyo.ConcreteModel()
        model.I = pyo.Set(initialize=[("ROOT", 0), ("ROOT", 1)], dimen=2)
        model.W = pyo.Param(
            model.I, mutable=True,
            initialize={("ROOT", 0): weights[0], ("ROOT", 1): weights[1]},
        )
        node = SimpleNamespace(name="ROOT")
        self._model = model
        self._mpisppy_node_list = [node]
        self._mpisppy_data = SimpleNamespace(
            nlens={"ROOT": 2},
            nonant_indices={("ROOT", 0): object(), ("ROOT", 1): object()},
            prob_coeff={"ROOT": probability},
        )
        self._mpisppy_model = SimpleNamespace(W=model.W)


@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                    reason="serial projection test")
def test_projection_removes_nodewise_probability_weighted_mean():
    scenarios = {
        "s0": _Scenario(0.25, [1.0, 4.0]),
        "s1": _Scenario(0.75, [5.0, -2.0]),
    }
    opt = SimpleNamespace(
        local_scenarios=scenarios,
        comms={"ROOT": MPI.COMM_WORLD},
    )

    removed = project_W_to_zero_sum(opt)

    np.testing.assert_allclose(removed["ROOT"], [4.0, -0.5])
    np.testing.assert_allclose(Wbar_by_node(opt)["ROOT"], [0.0, 0.0],
                               atol=1e-15)
