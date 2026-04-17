###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Drop-in sitecustomize that auto-starts coverage in every Python process.

This is used to capture coverage from MPI worker processes spawned by mpiexec.
It is activated by setting the environment variable:

    COVERAGE_PROCESS_START=.coveragerc

and adding this file's directory to PYTHONPATH (or sys.path).
"""
import coverage
coverage.process_startup()
