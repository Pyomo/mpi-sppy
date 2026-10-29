###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Focused tests for APH hub behavior that does not require MPI windows."""

from types import SimpleNamespace
from unittest import mock

from mpisppy.cylinders.hub import APHHub


def test_aph_hub_finalize_passes_extensions_to_post_loops():
    extensions = object()
    post_loops = mock.Mock(return_value=123.0)
    hub = object.__new__(APHHub)
    hub.opt = SimpleNamespace(
        extensions=extensions,
        post_loops=post_loops,
    )

    assert hub.finalize() == 123.0
    post_loops.assert_called_once_with(extensions)
