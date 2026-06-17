###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Apportion a fixed pool of MPI ranks across cylinders by target ratio.

Supports flexible (per-cylinder) rank counts for WheelSpinner.  See
``doc/designs/flexible_rank_assignments.md`` (the "User Interface"
section) for the algorithm and its rationale.
"""

import math


def apportion_ranks(ratios, n_proc):
    """Apportion ``n_proc`` ranks across cylinders by target ratio.

    Uses the largest-remainder (Hare quota) method, then enforces a
    floor of one rank per cylinder.  The returned counts sum to exactly
    ``n_proc`` and every entry is at least 1, so every cylinder runs.

    Args:
        ratios: ordered sequence of positive target ratios, one per
            cylinder, in declaration order (the hub is conventionally
            first with ratio 1.0).  Only relative magnitudes matter.
        n_proc: total number of MPI ranks to distribute (> 0).

    Returns:
        list[int]: rank count per cylinder, in the same order as
        ``ratios``, summing to ``n_proc``, each at least 1.

    Raises:
        ValueError: if there are more cylinders than ranks (a floor of
            one each is then infeasible), or if any ratio is
            non-positive, or if ``n_proc`` is non-positive, or if no
            cylinders are given.
    """
    n_cyl = len(ratios)
    if n_cyl == 0:
        raise ValueError("apportion_ranks: need at least one cylinder")
    if n_proc <= 0:
        raise ValueError(f"apportion_ranks: n_proc must be positive (got {n_proc})")
    if any(r <= 0 for r in ratios):
        raise ValueError(f"apportion_ranks: ratios must be positive (got {list(ratios)})")
    if n_cyl > n_proc:
        raise ValueError(
            f"apportion_ranks: {n_cyl} cylinders cannot each receive at least "
            f"one rank from only {n_proc} ranks"
        )

    total = math.fsum(ratios)
    real_share = [r / total * n_proc for r in ratios]
    counts = [math.floor(x) for x in real_share]

    # Largest-remainder: hand out the leftover ranks to the biggest
    # fractional remainders, breaking ties by declaration order (lowest
    # index first).
    leftover = n_proc - sum(counts)
    if leftover > 0:
        by_remainder = sorted(
            range(n_cyl),
            key=lambda i: (real_share[i] - counts[i], -i),
            reverse=True,
        )
        for i in by_remainder[:leftover]:
            counts[i] += 1

    # Floor of one: move a rank from the currently-largest cylinder to a
    # cylinder stuck at zero, until none remain at zero.  This is always
    # feasible because n_cyl <= n_proc, so whenever a zero exists some
    # other cylinder holds at least two.
    while True:
        zeros = [i for i, c in enumerate(counts) if c == 0]
        if not zeros:
            break
        donor = max(range(n_cyl), key=lambda i: counts[i])
        counts[donor] -= 1
        counts[zeros[0]] += 1

    return counts


def cylinder_bases(rank_counts):
    """First global rank of each cylinder's contiguous block.

    The unequal-rank path lays cylinders out as contiguous global-rank
    blocks in declaration order: cylinder ``i`` owns global ranks
    ``[bases[i], bases[i] + rank_counts[i])``.

    Args:
        rank_counts: per-cylinder rank counts (e.g. from ``apportion_ranks``).

    Returns:
        list[int]: ``bases[i]`` is the lowest global rank in cylinder ``i``.
    """
    bases = []
    base = 0
    for rc in rank_counts:
        bases.append(base)
        base += rc
    return bases


def rank_to_cylinder(global_rank, rank_counts):
    """Locate a global rank within the contiguous-block cylinder layout.

    Args:
        global_rank: a rank in ``[0, sum(rank_counts))``.
        rank_counts: per-cylinder rank counts (e.g. from ``apportion_ranks``).

    Returns:
        tuple[int, int]: ``(cylinder_index, rank_within_cylinder)``, where
        ``rank_within_cylinder`` is this rank's position in its cylinder's
        ``cylinder_comm`` (i.e. its ``cylinder_rank``).

    Raises:
        ValueError: if ``global_rank`` is outside the apportioned range.
    """
    base = 0
    for cyl, rc in enumerate(rank_counts):
        if base <= global_rank < base + rc:
            return cyl, global_rank - base
        base += rc
    raise ValueError(
        f"rank_to_cylinder: global_rank {global_rank} is outside the "
        f"apportioned range [0, {base}) for rank_counts {list(rank_counts)}"
    )
