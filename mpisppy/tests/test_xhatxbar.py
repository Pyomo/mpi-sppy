###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from types import SimpleNamespace
from unittest import mock

from mpisppy.extensions.xhatxbar import XhatXbar


class _FakeSpcomm:
    def __init__(self):
        self.BestInnerBound = float("inf")
        self.last_ib_idx = None

    def InnerBoundUpdate(self, new_bound, char="*"):
        self.last_ib_idx = 0
        self.latest_ib_char = char
        return new_bound


def _tree(*nodes):
    return SimpleNamespace(nonleaves=[
        SimpleNamespace(name=name, scenfirst=first, scenlast=last)
        for name, first, last in nodes
    ])


def test_final_xbar_registers_incumbent_for_solution_writers():
    spcomm = _FakeSpcomm()
    opt = SimpleNamespace(
        cylinder_rank=0,
        n_proc=1,
        options={
            "verbose": False,
            "xhat_xbar_options": {"xhat_solver_options": {}},
        },
        scenario_names_to_rank={"ROOT": {"Scenario0": 0}},
        all_scenario_names=["Scenario0"],
        _scenario_tree=_tree(("ROOT", 0, 0)),
        first_stage_solution_available=False,
        tree_solution_available=False,
        spcomm=spcomm,
        disable_W_and_prox=mock.Mock(),
        reenable_W_and_prox=mock.Mock(),
    )
    extension = XhatXbar(opt)
    extension.xhat_tryit = mock.Mock(return_value=123.0)

    extension.post_everything()

    extension.xhat_tryit.assert_called_once_with(
        verbose=False, restore_nonants=False)
    assert opt.first_stage_solution_available
    assert opt.tree_solution_available
    assert spcomm.BestInnerBound == 123.0
    assert spcomm.last_ib_idx == 0
    assert spcomm.latest_ib_char == "E"


def test_multistage_diagnostic_writer_uses_scenario_in_each_node(tmp_path):
    def scenario(name, *node_names):
        nodes = [SimpleNamespace(
            name=node_name,
            nonant_vardata_list=[SimpleNamespace(
                name=f"{name}.{node_name}.x", _value=1.0)],
        ) for node_name in node_names]
        return SimpleNamespace(
            _mpisppy_node_list=nodes,
            _mpisppy_data=SimpleNamespace(
                nlens=dict.fromkeys(node_names, 1)),
        )

    all_scenario_names = ["Scenario0", "Scenario1"]
    spcomm = _FakeSpcomm()
    opt = SimpleNamespace(
        cylinder_rank=0,
        n_proc=1,
        options={
            "verbose": False,
            "xhat_xbar_options": {
                "xhat_solver_options": {},
                "csvname": str(tmp_path / "xhatxbar.csv"),
            },
        },
        # scen_names_to_ranks() deliberately includes every scenario for
        # every node when n_proc == 1; it cannot identify node membership.
        scenario_names_to_rank={
            node_name: dict.fromkeys(all_scenario_names, 0)
            for node_name in ("ROOT", "Node0", "Node1")
        },
        all_scenario_names=all_scenario_names,
        _scenario_tree=_tree(
            ("ROOT", 0, 1), ("Node0", 0, 0), ("Node1", 1, 1)),
        local_scenarios={
            "Scenario0": scenario("Scenario0", "ROOT", "Node0"),
            "Scenario1": scenario("Scenario1", "ROOT", "Node1"),
        },
        first_stage_solution_available=False,
        tree_solution_available=False,
        spcomm=spcomm,
        disable_W_and_prox=mock.Mock(),
        reenable_W_and_prox=mock.Mock(),
    )
    extension = XhatXbar(opt)
    extension.xhat_tryit = mock.Mock(return_value=123.0)

    extension.post_everything()

    assert (tmp_path / "xhatxbar.csv_ROOT_Scenario0.csv").exists()
    assert (tmp_path / "xhatxbar.csv_Node0_Scenario0.csv").exists()
    assert (tmp_path / "xhatxbar.csv_Node1_Scenario1.csv").exists()
