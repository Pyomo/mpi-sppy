###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import sys
from runtests.mpi import Tester
import os.path

if __name__ == '__main__':
    tester = Tester(os.path.join(os.path.abspath(__file__)), ".")
    tester.main(sys.argv[1:])
