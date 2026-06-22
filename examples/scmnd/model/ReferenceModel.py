# -*- coding: utf-8 -*-
###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
###############################################################################


#
# Imports
#

from pyomo.environ import *

#
# Model
#

model = AbstractModel()

#
# Parameters
#

#Nodes, i in N
model.NumNodes = Param(within = PositiveIntegers)
model.Nodes    = RangeSet(model.NumNodes)

#Arcs, j in E
model.NumArcs = Param(within = PositiveIntegers)
model.Arcs    = RangeSet(model.NumArcs)

#Arcs nodes assignment, j -> (i,i')
model.ArcsNodes = Param(model.Arcs, within= model.Nodes * model.Nodes)

def Arcs_out_init(model, node):
    for j in model.ArcsNodes:
        if model.ArcsNodes[j][0] == node:
            yield j

def Arcs_in_init(model, node):
    for j in model.ArcsNodes:
        if model.ArcsNodes[j][1] == node:
            yield j

model.NodesArcsOut = Set(model.Nodes, within=model.Arcs, initialize=Arcs_out_init)
model.NodesArcsIn = Set(model.Nodes, within=model.Arcs, initialize=Arcs_in_init)

#Commodities, k in K
model.NumCommodities = Param(within = PositiveIntegers)
model.Commodities    = RangeSet(model.NumCommodities)

#Demand of node i for commodity k
model.NodesCommoditiesDemand = Param(model.Nodes, model.Commodities, within = Reals, default = 0.0)

#Fixed cost of arc j
model.ArcsFixedCost = Param(model.Arcs, within = NonNegativeReals, default = 0.0) 

#Variable cost of arc j for commodity k
model.ArcsVariableCost = Param(model.Arcs, within = NonNegativeReals, default = 0.0)

#Capacity of arc j
model.ArcsCapacity = Param(model.Arcs, within = NonNegativeReals, default = 0.0)

#Penalty cost for unmet demand
model.PenaltyCost = Param(within = NonNegativeReals, default = 10000.0)

#
# Variables
#

model.DesignArcsVar = Var(model.Arcs, within = UnitInterval)

model.ArcsFlowVar   = Var(model.Arcs, model.Commodities, within = NonNegativeReals)

model.DummyArcsFlowVar = Var(model.Commodities, within = NonNegativeReals)

#
# Constraints
#

def demand_satisfaction_rule(model, i, k):
    if (model.NodesCommoditiesDemand[i,k] > 0.0):
        dummy_arc = model.DummyArcsFlowVar[k]
    elif (model.NodesCommoditiesDemand[i,k] < 0.0):
        dummy_arc = -model.DummyArcsFlowVar[k]
    else:
        dummy_arc = 0.0

    return (sum(model.ArcsFlowVar[j,k] for j in model.NodesArcsOut[i]) - sum(model.ArcsFlowVar[j,k] for j in model.NodesArcsIn[i]) + dummy_arc == model.NodesCommoditiesDemand[i,k])

model.DemandSatisfaction = Constraint(model.Nodes, model.Commodities, rule=demand_satisfaction_rule)

def arc_capacity_rule(model, j):
    return (sum(model.ArcsFlowVar[j,k] for k in model.Commodities) <= model.ArcsCapacity[j] * model.DesignArcsVar[j])

model.ArcsCapacityConstr = Constraint(model.Arcs, rule=arc_capacity_rule)

#
# Objective
#

def firststage_cost_rule(model):
    return sum(model.ArcsFixedCost[j] * model.DesignArcsVar[j] for j in model.Arcs)

model.FirstStageCost = Expression(rule=firststage_cost_rule)

def secondstage_cost_rule(model):
    return sum(model.ArcsVariableCost[j] * model.ArcsFlowVar[j,k] for j in model.Arcs for k in model.Commodities) + sum(model.PenaltyCost * model.DummyArcsFlowVar[k] for k in model.Commodities)

model.SecondStageCost = Expression(rule=secondstage_cost_rule)

model.TotalCost = Objective(expr= model.FirstStageCost + model.SecondStageCost, sense=minimize)

# # Istance creation function
# instance = model.create_instance('code_scmnd/Canad/C/c33.dat')
# # instance.pprint()
# solver = SolverFactory('xpress_persistent')
# solver.set_instance(instance)
# result = solver.solve(instance)
# # check the objective value
# print("Objective value: ", value(instance.TotalCost))
# for v in instance.component_objects(Var, active=True):
#     print("Variable",v)
#     for index in v:
#         print ("   ",index, value(v[index]))