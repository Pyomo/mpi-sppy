import gurobipy as gp
import farmer
from gurobipy import GRB
import numpy as np
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
from mpisppy.utils import config

farmerstream = np.random.RandomState()

def scenario_creator(scenario_name, use_integer=False, sense=GRB.MINIMIZE, crops_multiplier=1, num_scens=None, seedoffset=0):
    scennum = sputils.extract_num(scenario_name)
    basenames = ['BelowAverageScenario', 'AverageScenario', 'AboveAverageScenario']
    basenum = scennum % 3
    groupnum = scennum // 3
    scenname = basenames[basenum] + str(groupnum)

    farmerstream.seed(scennum + seedoffset)

    if sense not in [GRB.MINIMIZE, GRB.MAXIMIZE]:
        raise ValueError("Model sense Not recognized")
    
    model = gp.Model(scenname)
    
    crops = ["WHEAT", "CORN", "SUGAR_BEETS"]
    CROPS = [f"{crop}{i}" for i in range(crops_multiplier) for crop in crops]

    # Data
    TOTAL_ACREAGE = 500.0 * crops_multiplier

    def get_scaled_data(indict):
        outdict = {}
        for i in range(crops_multiplier):
            for crop in crops:
                outdict[f"{crop}{i}"] = indict[crop]
        return outdict

    PriceQuota = get_scaled_data({'WHEAT': 100000.0, 'CORN': 100000.0, 'SUGAR_BEETS': 6000.0})
    SubQuotaSellingPrice = get_scaled_data({'WHEAT': 170.0, 'CORN': 150.0, 'SUGAR_BEETS': 36.0})
    SuperQuotaSellingPrice = get_scaled_data({'WHEAT': 0.0, 'CORN': 0.0, 'SUGAR_BEETS': 10.0})
    CattleFeedRequirement = get_scaled_data({'WHEAT': 200.0, 'CORN': 240.0, 'SUGAR_BEETS': 0.0})
    PurchasePrice = get_scaled_data({'WHEAT': 238.0, 'CORN': 210.0, 'SUGAR_BEETS': 100000.0})
    PlantingCostPerAcre = get_scaled_data({'WHEAT': 150.0, 'CORN': 230.0, 'SUGAR_BEETS': 260.0})

    Yield = {
        'BelowAverageScenario': {'WHEAT': 2.0, 'CORN': 2.4, 'SUGAR_BEETS': 16.0},
        'AverageScenario': {'WHEAT': 2.5, 'CORN': 3.0, 'SUGAR_BEETS': 20.0},
        'AboveAverageScenario': {'WHEAT': 3.0, 'CORN': 3.6, 'SUGAR_BEETS': 24.0}
    }

    yield_vals = {crop: Yield[basenames[basenum]][crop.rstrip("0123456789")] + (farmerstream.rand() if groupnum != 0 else 0) for crop in CROPS}
    
    # Variables
    DevotedAcreage = model.addVars(CROPS, vtype=GRB.INTEGER if use_integer else GRB.CONTINUOUS, lb=0.0, ub=TOTAL_ACREAGE, name="DevotedAcreage")
    QuantitySubQuotaSold = model.addVars(CROPS, lb=0.0, name="QuantitySubQuotaSold")
    QuantitySuperQuotaSold = model.addVars(CROPS, lb=0.0, name="QuantitySuperQuotaSold")
    QuantityPurchased = model.addVars(CROPS, lb=0.0, name="QuantityPurchased")

    # Constraints
    model.addConstr(gp.quicksum(DevotedAcreage[crop] for crop in CROPS) <= TOTAL_ACREAGE, "TotalAcreage")

    for crop in CROPS:
        model.addConstr(CattleFeedRequirement[crop] <= yield_vals[crop] * DevotedAcreage[crop] + QuantityPurchased[crop] - QuantitySubQuotaSold[crop] - QuantitySuperQuotaSold[crop], f"CattleFeedReq_{crop}")
        model.addConstr(QuantitySubQuotaSold[crop] + QuantitySuperQuotaSold[crop] - (yield_vals[crop] * DevotedAcreage[crop]) <= 0.0, f"LimitAmountSold_{crop}")
        model.addConstr(QuantitySubQuotaSold[crop] <= PriceQuota[crop], f"EnforceQuota_{crop}")

    # Objective
    total_costs = gp.quicksum(PlantingCostPerAcre[crop] * DevotedAcreage[crop] for crop in CROPS)
    purchase_costs = gp.quicksum(PurchasePrice[crop] * QuantityPurchased[crop] for crop in CROPS)
    subquota_revenue = gp.quicksum(SubQuotaSellingPrice[crop] * QuantitySubQuotaSold[crop] for crop in CROPS)
    superquota_revenue = gp.quicksum(SuperQuotaSellingPrice[crop] * QuantitySuperQuotaSold[crop] for crop in CROPS)

    total_cost = total_costs + purchase_costs - subquota_revenue - superquota_revenue
    model.setObjective(total_cost, sense)

    model.optimize()

    gd = {
        "scenario": model,
        "nonants": {("ROOT", i): v for i, v in enumerate(DevotedAcreage.values())},
        "nonants_coeffs": {("ROOT", i): v.Obj for i, v in enumerate(DevotedAcreage.values())},
        "nonant_fixedness": {("ROOT", i): v.LB == v.UB for i, v in enumerate(DevotedAcreage.values())},
        "nonant_start": {("ROOT", i): v.Start for i, v in enumerate(DevotedAcreage.values())},
        "nonant_names": {("ROOT", i): v.VarName for i, v in enumerate(DevotedAcreage.values())},
        "probability": "uniform",
        "sense": sense,
        "BFs": None
    }

    return gd

