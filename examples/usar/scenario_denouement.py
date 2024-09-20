###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Provides a function which is called after optimization."""
import pyomo.environ as pyo


def scenario_denouement(
    rank: int, name: str, scenario: pyo.ConcreteModel
) -> None:
    """Does nothing (is a no-op).

    This function is called after optimization finishes.

    Args:
        rank: Unused.
        name: Unused.
        scenario: Unused.
    """
    pass
