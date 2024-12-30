###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# This is a program for testing and development
import os
import sys
import gams
import gamspy_base

this_dir = os.path.dirname(os.path.abspath(__file__))

gamspy_base_dir = gamspy_base.__path__[0]

w = gams.GamsWorkspace(working_directory=this_dir, system_directory=gamspy_base_dir)

#model = w.add_job_from_file("farmer_average_ph.gms")
#model = w.add_job_from_file("farmer_augmented.gms")
#model = w.add_job_from_file("farmer_linear_augmented.gms")
#model = w.add_job_from_file("farmer_average_ph_quadratic")
model = w.add_job_from_file("farmer_average_ph_linearized")


model.run(output=sys.stdout)
