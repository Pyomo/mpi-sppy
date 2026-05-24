###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Low-level multi-source RMA assembly test for the unequal-rank path.

This exercises the dangerous part of the communication-layer cut -- real
passive-target one-sided RMA reads assembled from several remote ranks via
an overlap map -- in isolation from the hub/spoke stack. It builds an
``SPWindow`` directly on ``COMM_WORLD``, has a "writer" cylinder publish
known per-scenario values, and has a smaller "reader" cylinder reconstruct
its own scenarios' values with ``compute_overlap_segments`` + partial
``SPWindow.get`` reads. A reader rank whose scenarios straddle a writer-rank
boundary must stitch the result from two writer buffers; that is the case the
plain single-source reader cannot handle and the multi-source path must.

Run with: ``mpiexec -np 6 python -m mpi4py -m pytest <thisfile>`` (or via the
straight_tests / coverage harness, which launches mpiexec for it).
"""

import unittest

import numpy as np

from mpisppy import MPI
from mpisppy.cylinders.spwindow import Field, SPWindow, padded_len_n_doubles
from mpisppy.cylinders.overlap_map import compute_overlap_segments

fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()
global_size = fullcomm.Get_size()


def _contiguous_slices(n_scen, n_proc):
    """Contiguous per-rank scenario partition, matching the formula used by
    sputils._ScenTree.scen_names_to_ranks (rank -> list of global scen idxs)."""
    avg = n_scen / n_proc
    return [list(range(int(i * avg), int((i + 1) * avg))) for i in range(n_proc)]


def _scen_value(scen, item):
    """A distinct, recognizable value for (global scenario, item) so a
    misrouted read is caught rather than silently plausible."""
    return 1000.0 * scen + item


@unittest.skipUnless(global_size == 6, "needs exactly 6 MPI ranks (4 writers + 2 readers)")
class TestMultiSourceAssembly(unittest.TestCase):
    """Ranks 0-3 are a 4-rank writer cylinder; ranks 4-5 a 2-rank reader
    cylinder. Both cover the same N_SCEN scenarios, split differently, so a
    reader rank reads from several writer ranks."""

    N_SCEN = 10
    WRITER_RANKS = 4   # global ranks 0..3
    READER_RANKS = 2   # global ranks 4..5

    def _run_one(self, items_per_scen_k):
        writer_slices = _contiguous_slices(self.N_SCEN, self.WRITER_RANKS)
        reader_slices = _contiguous_slices(self.N_SCEN, self.READER_RANKS)
        k = items_per_scen_k

        is_writer = global_rank < self.WRITER_RANKS

        # Every rank builds the (collective) window. Writers publish a
        # NONANTS_VALS-like field sized to their local scenarios; readers
        # publish nothing and only read.
        if is_writer:
            my_scen = writer_slices[global_rank]
            data_len = k * len(my_scen)
            logical_len = data_len  # data only; no id slot needed for this test
            padded = padded_len_n_doubles(logical_len)
            my_fields = {Field.NONANTS_VALS: (logical_len, padded)}
        else:
            my_fields = {}

        win = SPWindow(my_fields, fullcomm)

        if is_writer:
            padded = win.buffer_layout[Field.NONANTS_VALS][2]
            buf = np.full(padded, np.nan, dtype="d")
            for i, scen in enumerate(my_scen):
                for j in range(k):
                    buf[i * k + j] = _scen_value(scen, j)
            win.put(buf, Field.NONANTS_VALS)

        # Ensure all writer puts are visible before any reader get.
        fullcomm.Barrier()

        if not is_writer:
            reader_local = global_rank - self.WRITER_RANKS
            my_scen = reader_slices[reader_local]
            segments = compute_overlap_segments(
                my_scen, writer_slices, [k] * self.N_SCEN
            )
            # writer cylinder's global-rank base is 0
            local_buf = np.full(k * len(my_scen), np.nan, dtype="d")
            for seg in segments:
                dest = local_buf[seg.local_offset : seg.local_offset + seg.count]
                win.get(dest, seg.remote_rank, Field.NONANTS_VALS,
                        item_offset=seg.remote_offset, item_count=seg.count)

            expected = np.array(
                [_scen_value(scen, j) for scen in my_scen for j in range(k)],
                dtype="d",
            )
            np.testing.assert_array_equal(
                local_buf, expected,
                err_msg=f"{global_rank=} {k=} {my_scen=} {segments=}",
            )
            # A straddling reader rank must have used more than one segment.
            if reader_local == 0:
                self.assertGreater(
                    len(segments), 1,
                    "reader rank 0 spans a writer-rank boundary; expected a "
                    "multi-segment assembly",
                )

        fullcomm.Barrier()
        win.free()

    def test_assembly_one_item_per_scenario(self):
        self._run_one(items_per_scen_k=1)

    def test_assembly_multi_item_per_scenario(self):
        self._run_one(items_per_scen_k=3)


if __name__ == "__main__":
    unittest.main()
