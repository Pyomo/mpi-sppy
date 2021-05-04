import pyomo.environ as pe

import mpisppy.utils.sputils as sputils
from mpisppy.opt import ef, sc
import logging
from mpi4py import MPI


if MPI.COMM_WORLD.Get_rank() == 0:
    logging.basicConfig(level=logging.INFO)


"""
To run this example:

mpirun -np 3 python -m mpi4py schur_complement.py
"""


class Farmer(object):
    def __init__(self):
        self.crops = ['WHEAT', 'CORN', 'SUGAR_BEETS']
        self.total_acreage = 500
        self.PriceQuota = {'WHEAT': 100000.0, 'CORN': 100000.0, 'SUGAR_BEETS': 6000.0}
        self.SubQuotaSellingPrice = {'WHEAT': 170.0, 'CORN': 150.0, 'SUGAR_BEETS': 36.0}
        self.SuperQuotaSellingPrice = {'WHEAT': 0.0, 'CORN': 0.0, 'SUGAR_BEETS': 10.0}
        self.CattleFeedRequirement = {'WHEAT': 200.0, 'CORN': 240.0, 'SUGAR_BEETS': 0.0}
        self.PurchasePrice = {'WHEAT': 238.0, 'CORN': 210.0, 'SUGAR_BEETS': 100000.0}
        self.PlantingCostPerAcre = {'WHEAT': 150.0, 'CORN': 230.0, 'SUGAR_BEETS': 260.0}
        self.scenarios = ['BelowAverageScenario', 'AverageScenario', 'AboveAverageScenario']
        self.crop_yield = dict()
        self.crop_yield['BelowAverageScenario'] = {'WHEAT': 2.0, 'CORN': 2.4, 'SUGAR_BEETS': 16.0}
        self.crop_yield['AverageScenario'] = {'WHEAT': 2.5, 'CORN': 3.0, 'SUGAR_BEETS': 20.0}
        self.crop_yield['AboveAverageScenario'] = {'WHEAT': 3.0, 'CORN': 3.6, 'SUGAR_BEETS': 24.0}
        self.scenario_probabilities = dict()
        self.scenario_probabilities['BelowAverageScenario'] = 0.15
        self.scenario_probabilities['AverageScenario'] = 0.7
        self.scenario_probabilities['AboveAverageScenario'] = 0.15


def create_scenario(scenario_name: str, farmer: Farmer):
    m = pe.ConcreteModel()

    m.crops = pe.Set(initialize=farmer.crops)
    m.devoted_acreage = pe.Var(m.crops, bounds=(0, farmer.total_acreage))
    m.total_acreage_con = pe.Constraint(expr=sum(m.devoted_acreage.values()) <= farmer.total_acreage)

    m.QuantitySubQuotaSold = pe.Var(m.crops, bounds=(0.0, None))
    m.QuantitySuperQuotaSold = pe.Var(m.crops, bounds=(0.0, None))
    m.QuantityPurchased = pe.Var(m.crops, bounds=(0.0, None))

    def EnforceCattleFeedRequirement_rule(m, i):
        return (farmer.CattleFeedRequirement[i] <= (farmer.crop_yield[scenario_name][i] * m.devoted_acreage[i]) +
                m.QuantityPurchased[i] - m.QuantitySubQuotaSold[i] - m.QuantitySuperQuotaSold[i])
    m.EnforceCattleFeedRequirement = pe.Constraint(m.crops, rule=EnforceCattleFeedRequirement_rule)

    def LimitAmountSold_rule(m, i):
        return (m.QuantitySubQuotaSold[i] +
                m.QuantitySuperQuotaSold[i] -
                (farmer.crop_yield[scenario_name][i] * m.devoted_acreage[i]) <= 0.0)
    m.LimitAmountSold = pe.Constraint(m.crops, rule=LimitAmountSold_rule)

    def EnforceQuotas_rule(m, i):
        return 0.0, m.QuantitySubQuotaSold[i], farmer.PriceQuota[i]
    m.EnforceQuotas = pe.Constraint(m.crops, rule=EnforceQuotas_rule)

    obj_expr = sum(farmer.PurchasePrice[crop] * m.QuantityPurchased[crop] for crop in m.crops)
    obj_expr -= sum(farmer.SubQuotaSellingPrice[crop] * m.QuantitySubQuotaSold[crop] for crop in m.crops)
    obj_expr -= sum(farmer.SuperQuotaSellingPrice[crop] * m.QuantitySuperQuotaSold[crop] for crop in m.crops)
    obj_expr += sum(farmer.PlantingCostPerAcre[crop] * m.devoted_acreage[crop] for crop in m.crops)

    m.obj = pe.Objective(expr=obj_expr)

    m._mpisppy_probability = farmer.scenario_probabilities[scenario_name]
    sputils.attach_root_node(model=m,
                             firstobj=sum(farmer.PlantingCostPerAcre[crop] * m.devoted_acreage[crop] for crop in m.crops),
                             varlist=[m.devoted_acreage])

    return m


def solve_with_extensive_form():
    farmer = Farmer()
    options = dict()
    options['solver'] = 'gurobi_persistent'
    scenario_kwargs = dict()
    scenario_kwargs['farmer'] = farmer
    opt = ef.ExtensiveForm(options=options,
                           all_scenario_names=farmer.scenarios,
                           scenario_creator=create_scenario,
                           scenario_creator_kwargs=scenario_kwargs)
    results = opt.solve_extensive_form()
    opt.report_var_values_at_rank0()
    return opt


def solve_with_sc():
    farmer = Farmer()
    options = dict()
    scenario_kwargs = dict()
    scenario_kwargs['farmer'] = farmer
    opt = sc.SchurComplement(options=options,
                             all_scenario_names=farmer.scenarios,
                             scenario_creator=create_scenario,
                             scenario_creator_kwargs=scenario_kwargs)
    results = opt.solve()
    opt.report_var_values_at_rank0()
    return opt


if __name__ == '__main__':
    # solve_with_extensive_form()
    solve_with_sc()
