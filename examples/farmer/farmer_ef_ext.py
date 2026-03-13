###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

# Minimal EFExtension test wrapper for the mpi-sppy farmer example
#
# This module mirrors the public API of farmer.py by delegating all functions
# to farmer.py, but adds one extra CLI/config option (min-wheat) and defines
# an EFExtension that enforces a minimum *expected* wheat production in the EF.
###############################################################################

import pyomo.environ as pyo
import mpisppy.utils.cfg_vanilla as vanilla

# Import the base farmer module we are wrapping
import farmer as _farmer

# Import EFExtension base class
from mpisppy.extensions.extension import EFExtension


# -----------------------------------------------------------------------------
# Delegating wrappers (same functions as farmer.py)
# -----------------------------------------------------------------------------
def scenario_creator(*args, **kwargs):
    return _farmer.scenario_creator(*args, **kwargs)


def scenario_names_creator(*args, **kwargs):
    return _farmer.scenario_names_creator(*args, **kwargs)


def inparser_adder(cfg):
    # First, add all base farmer args (including num_scens requirement, etc.)
    _farmer.inparser_adder(cfg)

    # Add our EFExtension test arg
    cfg.add_to_config(
        "min_wheat",
        description="Minimum expected wheat production (in yield units consistent with model data).",
        domain=float,
        default=0.0,
    )


def kw_creator(cfg):
    return _farmer.kw_creator(cfg)


def sample_tree_scen_creator(*args, **kwargs):
    return _farmer.sample_tree_scen_creator(*args, **kwargs)


def scenario_denouement(*args, **kwargs):
    return _farmer.scenario_denouement(*args, **kwargs)


def ef_dict_callback(ef_dict, cfg):
    '''
    '''
    vanilla.ef_extension_adder(ef_dict, MinExpectedWheatEF)
    if ef_dict['extension_kwargs'] is None:
        ef_dict['extension_kwargs'] = dict()
    ef_dict['extension_kwargs']['min_wheat'] = cfg.min_wheat

    return ef_dict
# -----------------------------------------------------------------------------
# EFExtension: enforce a minimum expected wheat production
# -----------------------------------------------------------------------------
class MinExpectedWheatEF(EFExtension):
    """
    EFExtension that adds a constraint to the EF model enforcing:

        E[ total_wheat_production ] >= cfg.min_wheat

    where total_wheat_production in a scenario is:
        sum_{wheat crops} Yield[c] * DevotedAcreage[c]

    """

    def __init__(self, ef_obj, min_wheat):
        super().__init__(ef_obj)

        if self.ef.extension_kwargs is None:
            raise ValueError("min wheat option was not set properly")
        self.min_wheat = min_wheat 

    def pre_solve(self):

        min_wheat = self.min_wheat

        efm = self.ef.ef  # the underlying Pyomo EF model

        cross_scen_expr = 0.0
        for scen_name, scen in self.ef.scenarios():
            sprob = scen._mpisppy_probability

            wheat_crops = [c for c in scen.CROPS if c.rstrip("0123456789") == "WHEAT"]
            scen_prod = sum(scen.Yield[c] * scen.DevotedAcreage[c] for c in wheat_crops)
            cross_scen_expr += sprob * scen_prod

        # attach expectation constraint
        efm.MinExpectedWheatConstraint = pyo.Constraint(expr=cross_scen_expr >= float(min_wheat))

        if min_wheat is None or float(min_wheat) <= 0.0:
            # If not requested, do nothing.
            efm.MinExpectedWheatConstraint.deactivate()            
