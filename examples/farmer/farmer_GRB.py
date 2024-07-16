# The farmer's problem in gurobipy
#
# Reference:
#  John R. Birge and Francois Louveaux. Introduction to Stochastic Programming.

# Gurobi functionality
import gurobipy as gp
from gurobipy import GRB

# Create gurobi model
m = gp.Model("two_stage_farmer_model")

# Define the parameters (data)
crops = ["wheat", "corn", "beets"]

Total_Area = 500                                                    # acres
Planting_Cost = {"wheat": 150, "corn": 230, "beets": 260}           # $/acre  
Selling_Price = {"wheat": 170, "corn": 150, "beets": 36}            # $/acre  
Excess_Selling_Price = 10                                           # $/T 
Purchase_Price = {"wheat": 238, "corn": 210, "beets": 100}          # $/T 
Min_Requirement_Crops = {"wheat": 200, "corn": 240, "beets": 0}     # T
Beets_Quota = 6000                                                  # T
Random_Yield = {"wheat": 2.5, "corn": 3.0, "beets": 20.0}           # $/T


# Add vars

# 1st stage
# Area in acres devoted to each crop 
area = m.addVars(crops, lb=0, name="area")

# 2nd stage

# Tons of crop c sold under scenario s 
sell = m.addVars(crops, lb=0, name="sell")

# Tons of sugar beets sold in exess of the quota under scenario s
sell_excess = m.addVar(lb=0, name="sell")

# Tons of crop c bought under scenario s
buy = m.addVars(crops, lb=0, name="buy")

# Objective function
minmize_profit = (
    - Excess_Selling_Price * sell_excess
    - gp.quicksum(Selling_Price[c] * sell[c] - Purchase_Price[c] * buy[c] for c in crops)
    + gp.quicksum(Planting_Cost[c] * area[c] for c in crops)
    )

m.setObjective(minmize_profit, GRB.MINIMIZE)

# Constraints

# Constraint on the total area
m.addConstr(gp.quicksum(area[c] for c in crops) <= Total_Area, "totalArea")

# Constraint on the min required crops 
m.addConstrs((Random_Yield[c] * area[c] - sell[c] + buy[c] >= Min_Requirement_Crops[c] for c in crops), "requirement")

# Constraint on meeting quoata
m.addConstr(sell['beets'] <= Beets_Quota, "quota")

# Constraint on dealing with the excess of the beets
m.addConstr(sell['beets'] + sell_excess <= Random_Yield['beets'] * area['beets'])

m.optimize()

m.write('two_stage_farmer_model.lp')

"""
for v in m.getVars():
    print(f'{v.varName}: {v.x}')
print(f'Optimal objective value: {m.objVal}')
"""

