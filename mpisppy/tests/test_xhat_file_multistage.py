###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""End-to-end MPI round trip for the multi-stage xhat file.

Runs a real multistage aircond cylinder system, writes the incumbent's
whole nonant tree to one CSV via ``WheelSpinner.write_tree_nonants``, then
reads it back *by name* with ``sputils.read_nonant_tree_csv`` and asserts
the entire tree round-trips.

The point of running under MPI -- versus the serial unit tests in
``test_sputils.py`` / ``test_xhat_from_file.py`` -- is to exercise
``SPBase.gather_nonant_tree_to_rank0``'s cross-rank merge. At ``-np 4``
the two cylinders (PH hub, xhatshuffle spoke) get two ranks each, so the
writing cylinder's scenarios are split across ranks and a single rank
holds only some second-stage subtrees. The assembled file containing
*every* non-leaf node therefore proves the gather stitched the subtrees
together across ranks, and that the per-node agreement check accepted the
shared ROOT (a disagreement would have raised inside the write).

mpiexec -np 4 python -m mpi4py -m pytest mpisppy/tests/test_xhat_file_multistage.py
"""

import os
import tempfile
import unittest

import numpy as np
from mpi4py import MPI

from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.sputils as sputils
import mpisppy.tests.examples.aircond as aircond
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver

comm = MPI.COMM_WORLD

solver_available, solver_name, _, _ = get_solver()
# The xhatshuffle stage-2 EF needs a non-persistent solver.
_, np_solver_name, _, _ = get_solver(persistent_OK=False)

# 3-stage tree: ROOT plus 4 second-stage (non-leaf) nodes; the stage-3
# leaves carry no nonants. At -np 4 the writing cylinder has 2 ranks, so
# each holds two whole subtrees -- the cross-rank-merge case we want.
BRANCHING_FACTORS = [4, 4]


def _cfg():
    cfg = config.Config()
    cfg.multistage()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.xhatshuffle_args()
    aircond.inparser_adder(cfg)
    cfg.solver_name = np_solver_name
    cfg.default_rho = 1.0
    cfg.max_iterations = 10
    cfg.branching_factors = BRANCHING_FACTORS
    cfg.max_solver_threads = 2
    return cfg


def _build_dicts(cfg):
    num_scens = int(np.prod(BRANCHING_FACTORS))
    all_scenario_names = aircond.scenario_names_creator(num_scens)
    all_nodenames = sputils.create_nodenames_from_branching_factors(
        BRANCHING_FACTORS)
    kwargs = aircond.kw_creator(cfg)
    beans = (cfg, aircond.scenario_creator, aircond.scenario_denouement,
             all_scenario_names)

    hub_dict = vanilla.ph_hub(
        *beans, scenario_creator_kwargs=kwargs, all_nodenames=all_nodenames)
    spoke = vanilla.xhatshuffle_spoke(
        *beans, scenario_creator_kwargs=kwargs, all_nodenames=all_nodenames)
    spoke["opt_kwargs"]["options"]["stage2_ef_solver_name"] = np_solver_name
    spoke["opt_kwargs"]["options"]["branching_factors"] = BRANCHING_FACTORS
    return hub_dict, [spoke], kwargs


@unittest.skipUnless(comm.size == 4, "needs exactly 4 MPI ranks")
@unittest.skipUnless(solver_available, "no solver is available")
class TestXhatFileMultistageRoundTrip(unittest.TestCase):

    def test_write_then_read_round_trip(self):
        cfg = _cfg()
        hub_dict, spoke_list, kwargs = _build_dicts(cfg)
        wheel = WheelSpinner(hub_dict, spoke_list)
        wheel.spin()

        # A path all ranks agree on (only the winner's rank 0 actually writes).
        if comm.rank == 0:
            path = os.path.join(tempfile.gettempdir(),
                                f"xhat_ms_roundtrip_{os.getpid()}.csv")
        else:
            path = None
        path = comm.bcast(path, root=0)

        wheel.write_tree_nonants(path)
        comm.Barrier()

        # Rank 0 reads it back by name and checks the whole tree is present.
        if comm.rank == 0:
            try:
                self.assertTrue(os.path.exists(path),
                                "write_tree_nonants produced no file")
                # Build every scenario to recover, per node, the node-local
                # nonant variable names (the order read_nonant_tree_csv wants).
                order = {}
                for sname in aircond.scenario_names_creator(
                        int(np.prod(BRANCHING_FACTORS))):
                    s = aircond.scenario_creator(sname, **kwargs)
                    for node in s._mpisppy_node_list:
                        order.setdefault(
                            node.name,
                            [v.name for v in node.nonant_vardata_list])

                cache = sputils.read_nonant_tree_csv(path, order)

                # Every non-leaf node (ROOT + the 4 second-stage nodes) must be
                # present, even though any single writing rank held only some
                # subtrees -- this is the cross-rank-merge assertion.
                self.assertEqual(set(cache), set(order))
                self.assertEqual(len(cache), 1 + BRANCHING_FACTORS[0])
                for ndn, arr in cache.items():
                    self.assertTrue(np.all(np.isfinite(arr)),
                                    f"non-finite values for node {ndn}")
            finally:
                if os.path.exists(path):
                    os.remove(path)


if __name__ == "__main__":
    unittest.main()
