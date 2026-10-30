###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Focused tests for the asynchronous listener utility."""

from unittest import mock

from mpisppy.utils.listener_util.listener_util import Synchronizer


def test_async_run_joins_worker_before_returning():
    def work():
        pass

    synchronizer = object.__new__(Synchronizer)
    synchronizer.asynch = True
    synchronizer.work_fct = work
    synchronizer._rank = 0

    worker = mock.Mock()
    listener = mock.Mock()
    with mock.patch(
            "mpisppy.utils.listener_util.listener_util.threading.Thread",
            side_effect=[worker, listener]):
        synchronizer.run(args=[], kwargs=None)

    worker.start.assert_called_once_with()
    listener.start.assert_called_once_with()
    listener.join.assert_called_once_with()
    worker.join.assert_called_once_with()
