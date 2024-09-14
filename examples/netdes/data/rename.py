###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import subprocess

piper = subprocess.Popen(['ls'], 
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
stdout, stderr = piper.communicate()
fnames = str(stdout).split('\\n')[1:-1]

for fname in fnames:
    row = fname.replace('.txt', '').split('-')[2:]
    nnodes, nscens, density, idn = row[0], row[1], row[3], row[4]
    idn = idn[1:]
    density = 'L' if density == '30' else 'H'
    new_fname = '-'.join(['network', nnodes, nscens, density, idn])
    cmd = ['cp', fname, new_fname + '.dat']

