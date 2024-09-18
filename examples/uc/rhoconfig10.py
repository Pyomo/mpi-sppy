###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
from pyomo.environ import value

# TBD - we can't get the rho scale factor into the callback easily, so we hard-code for now.
rho_scale_factor = 1.0

def ph_rhosetter_callback(ph, scenario_tree, scenario):

   root_node = scenario_tree.findRootNode()

   scenario_instance = scenario._instance

   symbol_map = scenario_instance._ScenarioTreeSymbolMap

   # sorting to allow for sane verbose output...
   for b in sorted(scenario_instance.Buses):

       for t in sorted(scenario_instance.TimePeriods):

           for g in sorted(scenario_instance.ThermalGeneratorsAtBus[b]):

               min_power = value(scenario_instance.MinimumPowerOutput[g])
               max_power = value(scenario_instance.MaximumPowerOutput[g])
               avg_power = min_power + ((max_power - min_power) / 2.0)

               min_cost = value(scenario_instance.MinimumProductionCost[g])

               avg_cost = scenario_instance.ComputeProductionCosts(scenario_instance, g, t, avg_power) + min_cost
               
               rho = rho_scale_factor * avg_cost

               ph.setRhoOneScenario(root_node, 
                                    scenario, 
                                    symbol_map.getSymbol(scenario_instance.UnitOn[g,t]), 
                                    rho)
