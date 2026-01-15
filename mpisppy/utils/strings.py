###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from collections.abc import Iterable


def nice_join(
    seq: Iterable,
    separator: str = ", ",
    conjunction: str | None = "or",
    warp_in_single_quote: bool = False,
) -> str:
    """Joins items of a sequence into English phrases using the representation of the items and
    a given conjunction between the last two elements. If the sequence is an Iterable of strings and
    the conjunction is None, this function reproduces the behavior of the built-in-function `join`.

    Args:
        seq (Iterable): a sequence of strings to join
        separator (str, ", "): a delimiter to use between items
        conjunction (str | None, "or"): a conjunction to use between the last two items, or None to
            reproduce basic join behavior
        wrap_in_single_quote (bool, False): if True, all items of the sequence are warped in extra
            single quotes

    Returns:
        str: a joined string

    Examples:

        >>> from mpisppy.utils import nice_join

        >>> nice_join(["a", "b", "c"])
        "a, b or c"
    """
    q = "'" if warp_in_single_quote else ""
    seq = [f"{q}{x!s}{q}" for x in seq]

    if len(seq) <= 1 or conjunction is None:
        return separator.join(seq)
    else:
        return f"{separator.join(seq[:-1])} {conjunction} {seq[-1]}"


def scenario_names_creator(
    n: int, prefix: str = "scenario", separator: str = "", start: int | None = 0
) -> list[str]:
    """Creates a list of scenario names using the pattern `{prefix}{separator}{number}` for a single
    scenario name. The scenario names are distinguished using consecutive numbers.

    Args:
        n (int): number of wanted scenarios names
        prefix (str, "scenario"): string to start the scenario name
        separator (str, ""): string to separate the prefix from the consecutive number
        start (int | None, 0): number to use for the first scenario name

    Examples:

        >>> from mpisppy.utils import scenario_names_creator

        >>> scenario_names_creator(3, prefix="scen", separator="_", start=1)
        ["scen_1", "scen_2", "scen_3"]
    """
    # to avoid migration errors from earlier implementations start is also allowed to be None
    if start is None:
        start = 0
    return [f"{prefix}{separator}{i}" for i in range(start, start + n)]
