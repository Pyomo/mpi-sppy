###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Overlap maps for reading a local-sized field across cylinders that have
different MPI rank counts.

When two cylinders hold the same scenarios but split them over a different
number of ranks, a reader rank's local buffer for a per-scenario field is
backed by one or more *segments* living in remote ranks' buffers.  An
``OverlapSegment`` records, for one such segment, which remote rank holds
it and the matching offsets/length (in field *items*, e.g. nonants) in the
remote and local buffers.

This module is deliberately free of MPI and Pyomo: it consumes the
contiguous per-rank scenario partition (the ``slices`` produced by
``_ScenTree.scen_names_to_ranks``) plus a per-scenario item count, so it
can be unit tested directly.  See
``doc/designs/flexible_rank_assignments.md`` (the "Overlap Maps" section).
"""

from dataclasses import dataclass


@dataclass
class OverlapSegment:
    """One contiguous run of a local field buffer sourced from a single
    remote rank's buffer.  All offsets/lengths are in field items."""
    remote_rank: int     # rank in the peer cylinder to read from
    remote_offset: int   # item offset of this run within the remote buffer
    local_offset: int    # item offset of this run within the local buffer
    count: int           # number of items in this run


def compute_overlap_segments(local_scen_idxs, remote_slices, items_per_scen):
    """Segments mapping a local rank's field buffer onto remote buffers.

    Args:
        local_scen_idxs: ordered global scenario indices held by this
            (local) rank -- i.e. ``local_slices[local_rank]``.
        remote_slices: the peer cylinder's partition, ``rank -> ordered
            list of global scenario indices`` (the ``slices`` output of
            ``scen_names_to_ranks`` for that cylinder).  Determines both
            which remote rank holds each scenario and the scenario's
            offset within that rank's buffer.
        items_per_scen: sequence indexed by global scenario index giving
            the number of field items each scenario contributes (e.g.
            ``len(nonant_indices)``).  Uniform two-stage problems pass a
            constant for every scenario; multistage problems may vary.

    Returns:
        list[OverlapSegment]: segments covering the local buffer in order,
        with contiguous same-rank runs coalesced.  When both cylinders
        have the same rank count this degenerates to a single identity
        segment ``(remote_rank=local_rank, remote_offset=0,
        local_offset=0, count=<all items>)``.
    """
    # Invert remote_slices: global scenario index -> (remote rank, item
    # offset of that scenario within the remote rank's buffer).
    remote_rank_of_scen = {}
    remote_offset_of_scen = {}
    for rank, scen_idxs in enumerate(remote_slices):
        offset = 0
        for scen in scen_idxs:
            remote_rank_of_scen[scen] = rank
            remote_offset_of_scen[scen] = offset
            offset += items_per_scen[scen]

    segments = []
    cur = None
    local_offset = 0
    for scen in local_scen_idxs:
        rank = remote_rank_of_scen[scen]
        r_off = remote_offset_of_scen[scen]
        n = items_per_scen[scen]
        # Coalesce only while we stay on the same remote rank AND the
        # remote items remain contiguous with the run so far.
        if cur is not None and cur.remote_rank == rank \
                and cur.remote_offset + cur.count == r_off:
            cur.count += n
        else:
            if cur is not None:
                segments.append(cur)
            cur = OverlapSegment(remote_rank=rank, remote_offset=r_off,
                                 local_offset=local_offset, count=n)
        local_offset += n
    if cur is not None:
        segments.append(cur)
    return segments
