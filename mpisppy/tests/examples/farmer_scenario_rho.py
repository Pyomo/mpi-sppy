###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""farmer with a rho that grows with the scenario number.

With rho different in every scenario PH no longer keeps sum_s p_s W_s = 0,
which a checkpoint of a dual cylinder must not mistake for a damaged file.
"""

import mpisppy.utils.sputils as sputils
from mpisppy.tests.examples.farmer import (  # noqa: F401
    inparser_adder,
    kw_creator,
    sample_tree_scen_creator,
    scenario_creator,
    scenario_denouement,
    scenario_names_creator,
)


def _rho_setter(scen):
    k = sputils.extract_num(scen.name)
    return [(id(v), 1.0 + 0.5 * k) for v in scen.DevotedAcreage.values()]
