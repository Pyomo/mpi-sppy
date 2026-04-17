###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Farmer variant with a deliberate nonant ordering bug for testing.

For scenario scen2 the nonant variable list is reversed, which should be
caught by the nonant name validation check in spbase.py.

Usage (should raise RuntimeError):
    python -m mpisppy.generic_cylinders --module-name bad_farmer \
        --num-scens 3 --EF --EF-solver-name cplex
"""
import examples.farmer.farmer as farmer
import mpisppy.utils.sputils as sputils


def scenario_creator(scenario_name, **kwargs):
    model = farmer.scenario_creator(scenario_name, **kwargs)
    scennum = sputils.extract_num(scenario_name)
    if scennum == 2:
        # Deliberately reverse the nonant vardata list to create a mismatch.
        # This simulates a bug where the scenario creator provides nonant
        # variables in a different order for different scenarios.
        model._mpisppy_node_list[0].nonant_vardata_list.reverse()
    return model


def scenario_names_creator(num_scens, start=None):
    return farmer.scenario_names_creator(num_scens, start=start)


def inparser_adder(cfg):
    return farmer.inparser_adder(cfg)


def kw_creator(cfg):
    return farmer.kw_creator(cfg)


def scenario_denouement(rank, scenario_name, scenario):
    pass
