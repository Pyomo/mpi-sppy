###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import sys


begin_str = "total active prox cuts: "
max_val = 0
total_lines = 0
total_cuts = 1
with open(sys.argv[1], "r") as f:
    for line in f:
        if begin_str in line:
            val = int(line[len(begin_str) :])
            total_cuts += val
            total_lines += 1
            if val > max_val:
                max_val = val

print(f"max: {max_val}, mean: {total_cuts / total_lines}")
