#ReferenceModel for full set of scenarios for APL1P; May 2021
#We use costs from Bailey, Jensen and Morton, Response Surface Analysis of Two-Stage Stochastic Linear Programming with Recourse
#(costs are 10x higher than in the original [Infanger 1992] paper)

import pyomo.environ as pyo
import numpy as np
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgamator as amalgamator

# Use this random stream:
apl1pstream = np.random.RandomState()

def APL1P_model_creator(seed):
    apl1pstream.seed(seed)
    random_array = apl1pstream.rand(6) #We only use 5 random numbers
    
    #
    # Model
    #
    
    model = pyo.ConcreteModel()
    
    #
    # Parameters
    #
    
    # generator
    model.G = [1,2]
    
    # Demand level
    model.DL = [1,2,3]
    
    # Availability
    avail_outcome = ([1.,0.9,0.5,0.1],[1.,0.9,0.7,0.1,0.0])
    avail_probability = ([0.2,0.3,0.4,0.1],[0.1,0.2,0.5,0.1,0.1])
    avail_cumprob = (np.cumsum(avail_probability[0]),np.cumsum(avail_probability[1]))
    assert(max(avail_cumprob[0])==1.0 and max(avail_cumprob[1])==1.0)
    def availability_init(m,g):
        rd = random_array[g]
        i = np.searchsorted(avail_cumprob[g-1],rd)
        return avail_outcome[g-1][i]
    
    model.Availability = pyo.Param(model.G, within=pyo.NonNegativeReals,
                               initialize=availability_init)
    
    # Min Capacity
    cmin_init = 1000
    model.Cmin = pyo.Param(model.G, within=pyo.NonNegativeReals, initialize=cmin_init)
    
    
    # Investment, aka Capacity costs
    invest = np.array([4.,2.5])
    def investment_init(m,g):
        return(invest[g-1])

    model.Investment = pyo.Param(model.G, within=pyo.NonNegativeReals,
                             initialize=investment_init)
    
    # Operating Cost
    op_cost = np.array([[4.3,2.0,0.5],[8.7,4.0,1.0]])
    def operatingcost_init(m,g,dl):
        return(op_cost[g-1,dl-1])
    
    model.OperatingCost = pyo.Param(model.G, model.DL, within=pyo.NonNegativeReals,
                                initialize = operatingcost_init)
    
    # Demand
    demand_outcome = [900,1000,1100,1200]
    demand_prob = [.15,.45,.25,.15]
    demand_cumprob = np.cumsum(demand_prob)
    assert(max(demand_cumprob) == 1.0)
    def demand_init(m,dl):
        rd = random_array[2+dl]
        i = np.searchsorted(demand_cumprob,rd)
        return demand_outcome[i]
    
    model.Demand = pyo.Param(model.DL, within=pyo.NonNegativeReals,
                         initialize=demand_init)
    
    # Cost of unserved demand
    unserved_cost =10.0
    model.CostUnservedDemand = pyo.Param(model.DL, within=pyo.NonNegativeReals,
                                     initialize=unserved_cost)
    
    #
    # Variables
    #
    
    # Capacity of generators
    model.CapacityGenerators = pyo.Var(model.G, domain=pyo.NonNegativeReals)
    
    # Operation level
    model.OperationLevel = pyo.Var(model.G, model.DL, domain=pyo.NonNegativeReals)
    
    # Unserved demand
    model.UnservedDemand = pyo.Var(model.DL, domain=pyo.NonNegativeReals)
    
    
    #
    # Constraints
    #
    
    # Minimum capacity
    def MinimumCapacity_rule(model, g):
        return model.CapacityGenerators[g] >= model.Cmin[g]
    
    model.MinimumCapacity = pyo.Constraint(model.G, rule=MinimumCapacity_rule)
    
    # Maximum operating level
    def MaximumOperating_rule(model, g):
        return sum(model.OperationLevel[g, dl] for dl in model.DL) <= model.Availability[g] * model.CapacityGenerators[g]
    
    model.MaximumOperating = pyo.Constraint(model.G, rule=MaximumOperating_rule)
    
    # Satisfy demand
    def SatisfyDemand_rule(model, dl):
        return sum(model.OperationLevel[g, dl] for g in model.G) + model.UnservedDemand[dl] >= model.Demand[dl]
    
    model.SatisfyDemand = pyo.Constraint(model.DL, rule=SatisfyDemand_rule)
    
    #
    # Stage-specific cost computations
    #
    
    def ComputeFirstStageCost_rule(model):
        return sum(model.Investment[g] * model.CapacityGenerators[g] for g in model.G)
    
    model.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)
    
    
    def ComputeSecondStageCost_rule(model):
        expr = sum(
            model.OperatingCost[g, dl] * model.OperationLevel[g, dl] for g in model.G for dl in model.DL) + sum(
            model.CostUnservedDemand[dl] * model.UnservedDemand[dl] for dl in model.DL)
        return expr

    model.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)
    
    
    def total_cost_rule(model):
        return model.FirstStageCost + model.SecondStageCost
    
    model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    return(model)


def scenario_creator(sname, num_scens=None):
    scennum   = sputils.extract_num(sname)
    model = APL1P_model_creator(scennum)
    
    # Create the list of nodes associated with the scenario (for two stage,
    # there is only one node associated with the scenario--leaf nodes are
    # ignored).
    model._mpisppy_node_list = [
        scenario_tree.ScenarioNode(
            name="ROOT",
            cond_prob=1.0,
            stage=1,
            cost_expression=model.Total_Cost_Objective,
            nonant_list=[model.CapacityGenerators], 
            scen_model=model,
        )
    ]
    
    #Add the probability of the scenario
    if num_scens is not None :
        model._mpisppy_probability = 1/num_scens
    
    return(model)


#=========
def scenario_names_creator(num_scens,start=None):
    # (only for Amalgamator): return the full list of num_scens scenario names
    # if start!=None, the list starts with the 'start' labeled scenario
    if (start is None) :
        start=0
    return [f"scen{i}" for i in range(start,start+num_scens)]
        

#=========
def inparser_adder(inparser):
    # (only for Amalgamator): add command options unique to apl1p
    pass


#=========
def kw_creator(options):
    # (only for Amalgamator): linked to the scenario_creator and inparser_adder
    kwargs = {"num_scens" : options['num_scens'] if 'num_scens' in options else None,
              }
    return kwargs


#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass


#============================
def xhat_generator_apl1p(scenario_names, solvername="gurobi", solver_options=None):
    '''
    For sequential sampling.
    Takes scenario names as input and provide the best solution for the 
        approximate problem associated with the scenarios.
    Parameters
    ----------
    scenario_names: int
        Names of the scenario we use
    solvername: str, optional
        Name of the solver used. The default is "gurobi"
    solver_options: dict, optional
        Solving options. The default is None.

    Returns
    -------
    xhat: str
        A generated xhat, solution to the approximate problem induced by scenario_names.

    '''
    num_scens = len(scenario_names)
    
    ama_options = { "EF-2stage": True,
                    "EF_solver_name": solvername,
                    "EF_solver_options": solver_options,
                    "num_scens": num_scens,
                    "_mpisppy_probability": 1/num_scens,
                    }
    #We use from_module to build easily an Amalgamator object
    ama = amalgamator.from_module("mpisppy.tests.examples.apl1p",
                                  ama_options,use_command_line=False)
    #Correcting the building by putting the right scenarios.
    ama.scenario_names = scenario_names
    ama.run()
    
    # get the xhat
    xhat = sputils.nonant_cache_from_ef(ama.ef)

    return xhat

if __name__ == "__main__":
    #An example of sequential sampling for the APL1P model
    from mpisppy.confidence_intervals.seqsampling import SeqSampling
    optionsFSP = {'eps': 5.0,
                  'solvername': "gurobi_direct",
                  "c0":50,}
    apl1p_pb = SeqSampling("mpisppy.tests.examples.apl1p",
                            xhat_generator_apl1p, 
                            optionsFSP,
                            stopping_criterion="BPL",
                            stochastic_sampling=False)

    res = apl1p_pb.run()
    print(res)
