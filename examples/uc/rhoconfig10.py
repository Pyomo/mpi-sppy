# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
from pyomo.environ import *

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

               max_capacity = value(scenario_instance.MaximumPowerOutput[g])
               min_power = value(scenario_instance.MinimumPowerOutput[g])
               max_power = value(scenario_instance.MaximumPowerOutput[g])
               avg_power = min_power + ((max_power - min_power) / 2.0)

               min_cost = value(scenario_instance.MinimumProductionCost[g])

               fuel_cost = value(scenario_instance.FuelCost[g])

               avg_cost = scenario_instance.ComputeProductionCosts(scenario_instance, g, t, avg_power) + min_cost
               max_cost = scenario_instance.ComputeProductionCosts(scenario_instance, g, t, max_power) + min_cost
               
               rho = rho_scale_factor * avg_cost

               ph.setRhoOneScenario(root_node, 
                                    scenario, 
                                    symbol_map.getSymbol(scenario_instance.UnitOn[g,t]), 
                                    rho)
