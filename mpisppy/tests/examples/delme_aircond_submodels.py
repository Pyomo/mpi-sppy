#ReferenceModel for full set of scenarios for AirCond; June 2021


import pyomo.environ as pyo
import numpy as np
import time
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgamator as amalgamator
import argparse
from mpisppy import global_toc

# Use this random stream:
aircondstream = np.random.RandomState()

def demands_creator(sname,branching_factors,start_seed, mudev, sigmadev, 
                    starting_d=200,root_name="ROOT"):
    
    max_d = 400
    min_d = 0

    if branching_factors is None:
        raise RuntimeError("scenario_creator for aircond needs branching_factors")
    scennum   = sputils.extract_num(sname)
    # Find the right path and the associated seeds (one for each node) using scennum
    prod = np.prod(branching_factors)
    s = int(scennum % prod)
    d = starting_d
    demands = [d]
    nodenames = [root_name]
    for bf in branching_factors:
        assert prod%bf == 0
        prod = prod//bf
        nodenames.append(str(s//prod))
        s = s%prod
    
    stagelist = [int(x) for x in nodenames[1:]]
    for t in range(1,len(nodenames)):
        aircondstream.seed(start_seed+sputils.node_idx(stagelist[:t],branching_factors))
        d = min(max_d,max(min_d,d+aircondstream.normal(mudev,sigmadev)))
        demands.append(d)
    
    return demands,nodenames

def StageModel_creator(time, demand, last_stage=False, start_ups=None):
    model = pyo.ConcreteModel()
    model.T = [time]

    #Parameters
    #Demand
    model.Demand = demand
    #Inventory Cost
    model.InventoryCost = -0.8 if last_stage else 0.5
    #Regular Capacity
    model.Capacity = 200
    #Regular Production Cost
    model.RegularProdCost = 1
    #Overtime Production Cost
    model.OvertimeProdCost = 3

    model.start_ups = start_ups

    if model.start_ups:
        # Start-up Cost
        model.StartUpCost = 300# what should this be?

        # Negative Inventory Cost (for start-up cost)
        model.NegInventoryCost = -5
        model.max_T = 25 # this is for the start-up cost constraints, 
        # assume model will be less than 25 stages
        # maximum possible demand in a given stage
        model.M = 200 * model.max_T

    
    #Variables
    model.RegularProd = pyo.Var(domain=pyo.NonNegativeReals)
    
    model.OvertimeProd = pyo.Var(domain=pyo.NonNegativeReals)
    
    model.Inventory = pyo.Var(domain=pyo.NonNegativeReals)

    if model.start_ups:
        # start-up cost variable
        model.StartUp = pyo.Var(within=pyo.Binary)
        model.StartUpAux = pyo.Var(domain=pyo.NonNegativeReals)
    
    #Constraints
    def CapacityRule(m):
        return m.RegularProd<=m.Capacity
    model.MaximumCapacity = pyo.Constraint(rule=CapacityRule)

    if model.start_ups:
        model.RegStartUpConstraint = pyo.Constraint(
            expr=model.M*model.StartUp >= model.RegularProd + model.OvertimeProd)
        # for max function
        model.NegInventoryConstraint = pyo.ConstraintList()
        model.NegInventoryConstraint.add(model.StartUpAux >= model.InventoryCost*model.Inventory)
        model.NegInventoryConstraint.add(model.StartUpAux >= model.NegInventoryCost*model.Inventory)
    
    #Objective
    def stage_objective(m):
        if m.start_ups:
            return m.StartUp * m.StartUpCost +\
                m.RegularProdCost*m.RegularProd +\
                m.OvertimeProdCost*m.OvertimeProd +\
                m.StartUpAux
                #np.max([m.InventoryCost*m.Inventory, m.NegInventoryCost*m.Inventory])                
        else:
            return m.RegularProdCost*m.RegularProd +\
                m.OvertimeProdCost*m.OvertimeProd +\
                m.InventoryCost*m.Inventory


    model.StageObjective = pyo.Objective(rule=stage_objective,sense=pyo.minimize)
    
    return model

#Assume that demands has been drawn before
def aircond_model_creator(demands, start_ups=None):

    model = pyo.ConcreteModel()
    num_stages = len(demands)
    model.T = range(1,num_stages+1) #Stages 1,2,...,T
    model.max_T = 25
    if model.T[-1] > model.max_T:
        raise RuntimeError(
            'The number of stages exceeds the maximum expected for this model.')
    
    model.start_ups = start_ups

    #Parameters
    model.BeginInventory = 100
    
    #Creating stage models
    model.stage_models = {}
    for t in model.T:
        last_stage = (t==num_stages)
        model.stage_models[t] = StageModel_creator(t, demands[t-1],
            last_stage=last_stage, start_ups=model.start_ups)  

    #Constraints
    
    def material_balance_rule(m,t):
        if t == 1:
            return(m.BeginInventory+m.stage_models[t].RegularProd +\
                m.stage_models[t].OvertimeProd-m.stage_models[t].Inventory ==\
                m.stage_models[t].Demand)
        else:
            return(m.stage_models[t-1].Inventory +\
                m.stage_models[t].RegularProd+m.stage_models[t].OvertimeProd -\
                m.stage_models[t].Inventory ==\
                m.stage_models[t].Demand)
    model.MaterialBalance = pyo.Constraint(model.T, rule=material_balance_rule)

    #Objectives
    
    #Disactivating stage objectives
    for t in model.T:
        model.stage_models[t].StageObjective.deactivate()
    
    def stage_cost(stage_m):
        if model.start_ups:
            return stage_m.StartUp * stage_m.StartUpCost +\
                stage_m.RegularProdCost*stage_m.RegularProd +\
                stage_m.OvertimeProdCost*stage_m.OvertimeProd +\
                stage_m.StartUpAux
                #np.max([stage_m.InventoryCost*stage_m.Inventory, 
                #   stage_m.NegInventoryCost*stage_m.Inventory])
        else:
            return stage_m.RegularProdCost*stage_m.RegularProd +\
                stage_m.OvertimeProdCost*stage_m.OvertimeProd +\
                stage_m.InventoryCost*stage_m.Inventory
    def total_cost_rule(m):
        return(sum(stage_cost(m.stage_models[t]) for t in m.T))
    model.TotalCostObjective = pyo.Objective(rule = total_cost_rule,
                                             sense = pyo.minimize)
    
    #Last step
    for t in model.T:
        setattr(model, "stage_model_"+str(t), model.stage_models[t])
    
    return model

def MakeNodesforScen(model,nodenames,branching_factors,starting_stage=1):
    #Create all nonleaf nodes used by the scenario
    #Compatible with sample scenario creation
    TreeNodes = []
    for stage in model.T:

        nonant_list=[model.stage_models[stage].RegularProd,
                     model.stage_models[stage].OvertimeProd]
        if model.start_ups:
            nonant_list.append(model.stage_models[stage].StartUp)

        if stage ==1:
            ndn="ROOT"
            TreeNodes.append(scenario_tree.ScenarioNode(name=ndn,
                                                       cond_prob=1.0,
                                                       stage=stage,
                                                       cost_expression=model.stage_models[stage].StageObjective,
                                                       nonant_list=nonant_list,
                                                       scen_model=model,
                                                       nonant_ef_suppl_list = [model.stage_models[stage].Inventory],
                                                       )
                             )
        elif stage <=starting_stage:
            parent_ndn = ndn
            ndn = parent_ndn+"_0" #Only one node per stage before starting stage
            TreeNodes.append(scenario_tree.ScenarioNode(name=ndn,
                                                       cond_prob=1.0,
                                                       stage=stage,
                                                       cost_expression=model.stage_models[stage].StageObjective,
                                                       nonant_list=nonant_list,
                                                       scen_model=model,
                                                       nonant_ef_suppl_list = [model.stage_models[stage].Inventory],
                                                       parent_name = parent_ndn
                                                       )
                             )
        elif stage < max(model.T): #We don't add the leaf node
            parent_ndn = ndn
            ndn = parent_ndn+"_"+nodenames[stage-starting_stage]
            TreeNodes.append(scenario_tree.ScenarioNode(name=ndn,
                                                       cond_prob=1.0/branching_factors[stage-starting_stage-1],
                                                       stage=stage,
                                                       cost_expression=model.stage_models[stage].StageObjective,
                                                       nonant_list=nonant_list,
                                                       scen_model=model,
                                                       nonant_ef_suppl_list = [model.stage_models[stage].Inventory],
                                                       parent_name = parent_ndn
                                                       )
                             )
    return(TreeNodes)

        
def scenario_creator(sname, branching_factors=None, num_scens=None, mudev=0, sigmadev=40, start_seed=0, start_ups=None):
    if branching_factors is None:
        raise RuntimeError("scenario_creator for aircond needs branching_factors")

    demands,nodenames = demands_creator(sname, branching_factors, start_seed, mudev, sigmadev)
    
    model = aircond_model_creator(demands, start_ups=start_ups)
    
    if num_scens is not None:
        model._mpisppy_probability = 1/num_scens
    
    #Constructing the nodes used by the scenario
    model._mpisppy_node_list = MakeNodesforScen(model, nodenames, branching_factors)
    
    return(model)


def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    """ Create a scenario within a sample tree. Mainly for multi-stage and simple for two-stage.
    Args:
        sname (string): scenario name to be created
        stage (int >=1 ): for stages > 1, fix data based on sname in earlier stages
        sample_branching_factors (list of ints): branching factors for the sample tree
        seed (int): To allow randome sampling (for some problems, it might be scenario offset)
        given_scenario (Pyomo concrete model): if not None, use this to get data for ealier stages
        scenario_creator_kwargs (dict): keyword args for the standard scenario creator funcion
    Returns:
        scenario (Pyomo concrete model): A scenario for sname with data in stages < stage determined
                                         by the arguments
    """
    
    start_ups = scenario_creator_kwargs["start_ups"]
    # Finding demands from stage 1 to t
    if given_scenario is None:
        if stage == 1:
            past_demands = [200]
        else:
            raise RuntimeError(f"sample_tree_scen_creator for aircond needs a 'given_scenario' argument if the starting stage is greater than 1")
    else:
        past_demands = [given_scenario.stage_models[t].Demand for t in given_scenario.T if t<=stage]
    optional_things = ['mudev','sigmadev','start_ups']
    default_values = [0,40,False]
    for thing,value in zip(optional_things,default_values):
        if thing not in scenario_creator_kwargs:
            scenario_creator_kwargs[thing] = value
    
    #Finding demands for stages after t
    future_demands,nodenames = demands_creator(sname, sample_branching_factors, 
                                               start_seed = seed, 
                                               mudev = scenario_creator_kwargs['mudev'], 
                                               sigmadev = scenario_creator_kwargs['sigmadev'],
                                               starting_d=past_demands[stage-1],
                                               root_name='ROOT'+'_0'*(stage-1))
    
    demands = past_demands+future_demands[1:] #The demand at the starting stage is in both past and future demands
    
    model = aircond_model_creator(demands, start_ups=start_ups)
    
    model._mpisppy_probability = 1/np.prod(sample_branching_factors)
    
    #Constructing the nodes used by the scenario
    model._mpisppy_node_list = MakeNodesforScen(model, nodenames, sample_branching_factors,
                                                starting_stage=stage)
    
    return model
                

#=========
def scenario_names_creator(num_scens,start=None):
    # (only for Amalgamator): return the full list of num_scens scenario names
    # if start!=None, the list starts with the 'start' labeled scenario
    if (start is None) :
        start=0
    return [f"scen{i}" for i in range(start,start+num_scens)]
        

#=========
def inparser_adder(inparser):
    # (only for Amalgamator): add command options unique to aircond
    inparser.add_argument("--mu-dev",
                          help="average deviation of demand between two periods",
                          dest="mudev",
                          type=float,
                          default=0.)
    inparser.add_argument("--sigma-dev",
                          help="average standard deviation of demands between two periods",
                          dest="sigmadev",
                          type=float,
                          default=40.)
    inparser.add_argument("--with-start-ups",
                           help="Include start-up costs in model (this is a MIP)",
                           dest="start_ups",
                           action="store_true"
                           )
    inparser.add_argument("--start-seed",
                          help="random number seed (default 1134)",
                          dest="sigmadev",
                          type=int,
                          default=1134)
    inparser.set_defaults(start_ups=False)
    return inparser


#=========
def kw_creator(options):

    def _kwarg(option_name, default = None, arg_name=None):
        # options trumps args
        retval = options.get(option_name)
        if retval is not None:
            return retval
        args = options.get('args')
        aname = option_name if arg_name is None else arg_name
        retval = getattr(args, aname) if hasattr(args, aname) else None
        retval = default if retval is None else retval
        return retval

    # (only for Amalgamator): linked to the scenario_creator and inparser_adder
    # for confidence intervals, we need to see if the values are in args
    BFs = _kwarg("branching_factors")
    mudev = _kwarg("mudev", 0.)
    sigmadev = _kwarg("sigmadev", 40.)
    start_seed = _kwarg("start_seed", 1134)
    start_ups = _kwarg("start_ups", None)
    kwargs = {"num_scens" : options.get('num_scens', None),
              "branching_factors": BFs,
              "mudev": mudev,
              "sigmadev": sigmadev,
              "start_seed": start_seed,
              "start_ups": start_ups,
              }
    if kwargs["start_ups"] is None:
        raise ValueError("kw_creator called, but no value given for start_ups")
    return kwargs


#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass

#============================
def xhat_generator_aircond(scenario_names, solvername="gurobi", solver_options=None,
                           branching_factors=None, mudev = 0, sigmadev = 40,
                           start_ups=None, start_seed = 0):
    '''
    For sequential sampling.
    Takes scenario names as input and provide the best solution for the 
        approximate problem associated with the scenarios.
    Parameters
    ----------
    scenario_names: list of str
        Names of the scenario we use
    solvername: str, optional
        Name of the solver used. The default is "gurobi"
    solver_options: dict, optional
        Solving options. The default is None.
    branching_factors: list, optional
        Branching factors of the scenario 3. The default is [3,2,3] 
        (a 4 stage model with 18 different scenarios)
    mudev: float, optional
        The average deviation of demand between two stages; The default is 0.
    sigma_dev: float, optional
        The standard deviation from mudev for the demand difference between
        two stages. The default is 40.
    start_seed: int, optional
        The starting seed, used to create different sample scenario trees.
        The default is 0.

    Returns
    -------
    xhat: str
        A generated xhat, solution to the approximate problem induced by scenario_names.
        
    NOTE: This tool only works when the file is in mpisppy. In SPInstances, 
            you must change the from_module line.

    '''
    num_scens = len(scenario_names)
    
    ama_options = { "EF-mstage": True,
                    "EF_solver_name": solvername,
                    "EF_solver_options": solver_options,
                    "num_scens": num_scens,
                    "_mpisppy_probability": 1/num_scens,
                    "branching_factors":branching_factors,
                    "mudev":mudev,
                    "start_ups":start_ups,
                    "start_seed":start_seed,
                    "sigmadev":sigmadev
                    }
    #We use from_module to build easily an Amalgamator object
    ama = amalgamator.from_module("mpisppy.tests.examples.aircond_submodels",
                                  ama_options,use_command_line=False)
    #Correcting the building by putting the right scenarios.
    ama.scenario_names = scenario_names
    ama.verbose = False
    ama.run()
    
    # get the xhat
    xhat = sputils.nonant_cache_from_ef(ama.ef)

    return xhat

if __name__ == "__main__":
    # This __main__ is just for developers to use for quick tests
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver-name',
                        help="solver name (default 'gurobi_direct'",
                        default='gurobi_direct')
    parser.add_argument("--branching-factors",
                        help="Spaces delimited branching factors (default 3 3)",
                        dest="branching_factors",
                        nargs="*",
                        type=int,
                        default=[10,10])
    parser.add_argument("--with-start-ups",
                        help="Toggles start-up costs in the aircond model",
                        dest="start_ups",
                        action="store_true")
    parser.set_defaults(start_ups=False)
    parser.add_argument("--save-xhat",
                        help="Toggles file save",
                        dest="save_xhat",
                        action="store_true")
    parser.set_defaults(save_xhat=False)


    args=parser.parse_args()

    solver_name=args.solver_name
    start_ups=args.start_ups
    bfs = args.branching_factors
    save_xhat = args.save_xhat

    if start_ups:
        solver_options={"mipgap":0.015,'verbose':True,}
    else:
        solver_options=dict()

    num_scens = np.prod(bfs) #To check with a full tree
    ama_options = { "EF-mstage": True,
                    "EF_solver_name": solver_name,
                    "EF_solver_options":solver_options,
                    "num_scens": num_scens,
                    "branching_factors":bfs,
                    "mudev":0,
                    "sigmadev":40,
                    "start_ups":start_ups,
                    "start_seed":0
                    }
    refmodel = "mpisppy.tests.examples.aircond_submodels" # WARNING: Change this in SPInstances
    #We use from_module to build easily an Amalgamator object
    t0=time.time()
    ama = amalgamator.from_module(refmodel,
                                  ama_options,
                                  use_command_line=False)
    ama.run()
    print('start ups costs: ', start_ups)
    print('branching factors: ', bfs)
    print('run time: ', time.time()-t0)
    print(f"inner bound =", ama.best_inner_bound)
    print(f"outer bound =", ama.best_outer_bound)
    
    xhat = sputils.nonant_cache_from_ef(ama.ef)
    print('xhat_one = ', xhat['ROOT'])

    if save_xhat:
        bf_string = ''
        for bf in bfs:
            bf_string = bf_string + bf + '_'
        np.savetxt('aircond_start_ups='+str(start_ups)+bf_string+'zhatstar.txt', xhat['ROOT'])
   
        
        
