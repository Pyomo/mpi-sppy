# -*- coding: utf-8 -*-
###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
Created on Fri Feb  8 11:49:24 2013

@author: dgade
edited by dlw Novemeber 2016 so there is one model for all instances 
  and a litle cleanup
"""

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

#Potential facility location, index, j = 1,..,n
model.NumServers = Param(within = PositiveIntegers)
model.Servers    = RangeSet(model.NumServers)

#Clients, j = 1,..,m
model.NumClients = Param(within = PositiveIntegers)
model.Clients   = RangeSet(model.NumClients)

#Demand of client i
model.Demand    = Param(model.Clients, model.Servers, within = NonNegativeReals, default = 0.0)

#Fixed cost of Plant j
model.FixedCost = Param(model.Servers, within = NonNegativeReals, default = 0.0) #c_j in book

model.Capacity = Param(within = NonNegativeReals)

model.ClientPresent = Param(model.Clients, within = Binary, default = 1)

model.Revenue = Param(model.Clients,model.Servers, within = NonNegativeReals, default = 0.0)

model.Penalty     = Param(default = 1000.0)

#
# Variables
#

#model.FacilityOpen = Var(model.Servers, within = NonNegativeReals, bounds = (0,1))
model.FacilityOpen = Var(model.Servers, within = Binary)

#model.Allocation   = Var(model.Clients, model.Servers, within = NonNegativeReals,bounds=(0,1))
model.Allocation   = Var(model.Clients, model.Servers, within = Binary)

model.Dummy       = Var(model.Servers, within = NonNegativeReals)
#model.Dummy       = Var(model.Servers, within = NonNegativeIntegers)

#
# Constraints
#

def demand_constraint_rule(m,j):
    return sum(m.Demand[i,j] * m.Allocation[i,j] for i in m.Clients) - m.Dummy[j] <= m.Capacity*m.FacilityOpen[j]
model.DemandConstraint = Constraint(model.Servers, rule  = demand_constraint_rule)

def client_rule(m,i):
    return sum(m.Allocation[i,j] for j in m.Servers) == m.ClientPresent[i]
model.ClientConstraint = Constraint(model.Clients, rule = client_rule)

#
# Stage-specific cost computations
#

def first_stage_cost_rule(m):
    return summation(m.FixedCost, m.FacilityOpen)
model.FirstStageCost = Expression(rule = first_stage_cost_rule)

def second_stage_cost_rule(m):
    return m.Penalty * summation(m.Dummy) - summation(m.Revenue, m.Allocation)
model.SecondStageCost = Expression(rule = second_stage_cost_rule)

#
# Objective
#

def objective_rule(m):
    return m.FirstStageCost + m.SecondStageCost

model.MaxRevenue = Objective(rule = objective_rule, sense = minimize)

        
