#ReferenceModel for full set of scenarios for AirCond; June 2021


import pyomo.environ as pyo
import numpy as np
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgomator as amalgomator
from mpisppy import global_toc

# Use this random stream:
aircondstream = np.random.RandomState()

def demands_creator(sname,BFs,start_seed, mudev, sigmadev, 
                    starting_d=200,root_name="ROOT"):
    if BFs is None:
        raise RuntimeError("scenario_creator for aircond needs BFs")
    scennum   = sputils.extract_num(sname)
    # Find the right path and the associated seeds (one for each node) using scennum
    prod = np.prod(BFs)
    s = int(scennum % prod)
    d = starting_d
    demands = [d]
    nodenames = [root_name]
    for bf in BFs:
        assert prod%bf == 0
        prod = prod//bf
        nodenames.append(str(s//prod))
        s = s%prod
    
    stagelist = [int(x) for x in nodenames[1:]]
    for t in range(1,len(nodenames)):
        aircondstream.seed(start_seed+sputils.node_idx(stagelist[:t],BFs))
        d = min(400,max(0,d+aircondstream.normal(mudev,sigmadev)))
        demands.append(d)
    
    return demands,nodenames

def StageModel_creator(time, demand, last_stage=False):
    model = pyo.ConcreteModel()
    model.T = [time]
    #Parameters
    #Demand
    model.Demand = demand
    #Inventoy Cost
    model.InventoryCost = -0.8 if last_stage else 0.5
    #Regular Capacity
    model.Capacity = 200
    #Regular Production Cost
    model.RegularProdCost = 1
    #Overtime Production Cost
    model.OvertimeProdCost = 3

    
    #Variables
    model.RegularProd = pyo.Var(domain=pyo.NonNegativeReals)
    
    model.OvertimeProd = pyo.Var(domain=pyo.NonNegativeReals)
    
    model.Inventory = pyo.Var(domain=pyo.NonNegativeReals)
    
    #Constraints
    def CapacityRule(m):
        return m.RegularProd<=m.Capacity
    model.MaximumCapacity = pyo.Constraint(rule=CapacityRule)
    
    #Objective
    def stage_objective(m):
        return m.RegularProdCost*m.RegularProd +\
            m.OvertimeProdCost*m.OvertimeProd +\
                m.InventoryCost*m.Inventory
    model.StageObjective = pyo.Objective(rule=stage_objective,sense=pyo.minimize)
    
    return model

#Assume that demands has been drawn before
def aircond_model_creator(demands):
    model = pyo.ConcreteModel()
    num_stages = len(demands)
    model.T = range(1,num_stages+1) #Stages 1,2,...,T
    
    #Parameters
    model.BeginInventory = 100
    
    #Creating stage models
    model.stage_models = {}
    for t in model.T:
        last_stage = (t==num_stages)
        model.stage_models[t] = StageModel_creator(t, demands[t-1],last_stage)  

    #Constraints
    
    def material_balance_rule(m,t):
        if t == 1:
            return(m.BeginInventory+m.stage_models[t].RegularProd+m.stage_models[t].OvertimeProd-m.stage_models[t].Inventory == m.stage_models[t].Demand)
        else:
            return(m.stage_models[t-1].Inventory+m.stage_models[t].RegularProd+m.stage_models[t].OvertimeProd-m.stage_models[t].Inventory == m.stage_models[t].Demand)
    model.MaterialBalance = pyo.Constraint(model.T, rule=material_balance_rule)
    
    #Objectives
    
    #Disactivating stage objectives
    for t in model.T:
        model.stage_models[t].StageObjective.deactivate()
    
    def stage_cost(stage_m):
        return  stage_m.RegularProdCost*stage_m.RegularProd +\
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
        if stage ==1:
            ndn="ROOT"
            TreeNodes.append(scenario_tree.ScenarioNode(name=ndn,
                                                       cond_prob=1.0,
                                                       stage=stage,
                                                       cost_expression=model.stage_models[stage].StageObjective,
                                                       scen_name_list=None, # Not maintained
                                                       nonant_list=[model.stage_models[stage].RegularProd,
                                                                    model.stage_models[stage].OvertimeProd],
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
                                                       scen_name_list=None, # Not maintained
                                                       nonant_list=[model.stage_models[stage].RegularProd,
                                                                    model.stage_models[stage].OvertimeProd],
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
                                                       scen_name_list=None, # Not maintained
                                                       nonant_list=[model.stage_models[stage].RegularProd,
                                                                    model.stage_models[stage].OvertimeProd],
                                                       scen_model=model,
                                                       nonant_ef_suppl_list = [model.stage_models[stage].Inventory],
                                                       parent_name = parent_ndn
                                                       )
                             )
    return(TreeNodes)

        
def scenario_creator(sname, BFs=None, num_scens=None, mudev=0, sigmadev=40, start_seed=0):
    if BFs is None:
        raise RuntimeError("scenario_creator for aircond needs BFs")

    demands,nodenames = demands_creator(sname, BFs, start_seed, mudev, sigmadev)
    
    model = aircond_model_creator(demands)
    
    if num_scens is not None:
        model._mpisppy_probability = 1/num_scens
    
    #Constructing the nodes used by the scenario
    model._mpisppy_node_list = MakeNodesforScen(model, nodenames, BFs)
    
    return(model)

def sample_tree_scen_creator(sname,given_scenario=None,stage=None,sample_BFs=None, seed=None, **scenario_creator_kwargs):
    #For multistage confidence interval
    
    things = [stage,sample_BFs,seed]
    names = ["stage","sample_BFs","seed"]
    for i in range(len(things)):
        if things[i] is None:
            raise RuntimeError(f"sample_tree_scen_creator for aircond needs a {names[i]} argument.")
    
    #Finding demands from stage 1 to t
    if given_scenario is None:
        if stage == 1:
            past_demands = [200]
        else:
            raise RuntimeError(f"sample_tree_scen_creator for aircond needs a 'given_scenario' argument if the starting stage is greater than 1")
    else:
        past_demands = [given_scenario.stage_models[t].Demand for t in given_scenario.T if t<=stage]
    optional_things = ['mudev','sigmadev']
    default_values = [0,40]
    for thing,value in zip(optional_things,default_values):
        if thing not in scenario_creator_kwargs:
            scenario_creator_kwargs[thing] = value
    
    #Finding demands for stages after t
    future_demands,nodenames = demands_creator(sname, sample_BFs, 
                                               start_seed = seed, 
                                               mudev = scenario_creator_kwargs['mudev'], 
                                               sigmadev = scenario_creator_kwargs['sigmadev'],
                                               starting_d=past_demands[stage-1],
                                               root_name='ROOT'+'_0'*(stage-1))
    
    demands = past_demands+future_demands[1:] #The demand at the starting stage is in both past and future demands
    
    model = aircond_model_creator(demands)
    
    model._mpisppy_probability = 1/np.prod(sample_BFs)
    
    #Constructing the nodes used by the scenario
    model._mpisppy_node_list = MakeNodesforScen(model, nodenames, sample_BFs,
                                                starting_stage=stage)
    
    return model
                

#=========
def scenario_names_creator(num_scens,start=None):
    # (only for Amalgomator): return the full list of num_scens scenario names
    # if start!=None, the list starts with the 'start' labeled scenario
    if (start is None) :
        start=0
    return [f"scen{i}" for i in range(start,start+num_scens)]
        

#=========
def inparser_adder(inparser):
    # (only for Amalgomator): add command options unique to farmer
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

#=========
def kw_creator(options):
    # (only for Amalgomator): linked to the scenario_creator and inparser_adder
    kwargs = {"num_scens" : options['num_scens'] if 'num_scens' in options else None,
              "BFs" : options['BFs'] if 'BFs' in options else [3,2,3],
              "mudev" : options['mudev'] if 'mudev' in options else 0.,
              "sigmadev" : options['sigmadev'] if 'sigmadev' in options else 40.,
              "start_seed": options['start_seed'] if 'start_seed' in options else 0,
              }
    return kwargs


#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass
        
#============================
def xhat_generator_aircond(scenario_names, solvername="gurobi", solver_options=None,
                           BFs=[3,2,3], mudev = 0, sigmadev = 40, start_seed = 0):
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
    BFs: list, optional
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
                    "BFs":BFs,
                    "mudev":mudev,
                    "start_seed":start_seed,
                    "sigmadev":sigmadev
                    }
    #We use from_module to build easily an Amalgomator object
    ama = amalgomator.from_module("mpisppy.tests.examples.aircond_submodels",
                                  ama_options,use_command_line=False)
    #Correcting the building by putting the right scenarios.
    ama.scenario_names = scenario_names
    ama.verbose = False
    ama.run()
    
    # get the xhat
    xhat = sputils.nonant_cache_from_ef(ama.ef)

    return {'ROOT': xhat['ROOT']}
    

    
    

if __name__ == "__main__":
    bfs = [3,3,2]
    num_scens = np.prod(bfs) #To check with a full tree
    ama_options = { "EF-mstage": True,
                    "EF_solver_name": "gurobi_direct",
                    "num_scens": num_scens,
                    "_mpisppy_probability": 1/num_scens,
                    "BFs":bfs,
                    "mudev":0,
                    "sigmadev":80
                    }
    refmodel = "mpisppy.tests.examples.aircond_submodels" # WARNING: Change this in SPInstances
    # #We use from_module to build easily an Amalgomator object
    # ama = amalgomator.from_module(refmodel,
    #                               ama_options,use_command_line=False)
    # ama.run()
    # print(f"inner bound=", ama.best_inner_bound)
    # print(f"outer bound=", ama.best_outer_bound)
    
    # from mpisppy.confidence_intervals.mmw_ci import MMWConfidenceIntervals
    # options = ama.options
    # options['solver_options'] = options['EF_solver_options']
    # xhat = sputils.nonant_cache_from_ef(ama.ef)
   
    
    # num_batches = 10
    # batch_size = 100
    
    # mmw = MMWConfidenceIntervals(refmodel, options, xhat, num_batches,batch_size=batch_size,
    #                     verbose=False)
    # r=mmw.run(objective_gap=True)
    # print(r)
    
    #An example of sequential sampling for the aircond model
    from mpisppy.confidence_intervals.seqsampling import SeqSampling
    optionsBM =  { 'h':0.5,
                    'hprime':0.1, 
                    'eps':0.5, 
                    'epsprime':0.4, 
                    "p":0.2,
                    "q":1.2,
                    "solvername":"gurobi_direct",
                    "BFs": bfs}
    
    optionsFSP = {'eps': 50.0,
                  'solvername': "gurobi_direct",
                  "c0":50,
                  "BFs": bfs,
                  "xhat_gen_options":{'mudev':0, 'sigmadev':40},
                  'mudev':0,
                  'sigmadev':40
                  }
    aircondpb = SeqSampling("mpisppy.tests.examples.aircond_submodels",
                            xhat_generator_aircond, 
                            optionsFSP,
                            stopping_criterion="BPL",
                            stochastic_sampling=False,
                            solving_type="EF-mstage")

    res = aircondpb.run()
    print(res)
    
    #Doing replicates
    nrep = 10
    seed = 0
    res = []
    for k in range(nrep):
        aircondpb = SeqSampling("mpisppy.tests.examples.aircond_submodels",
                            xhat_generator_aircond, 
                            optionsFSP,
                                stopping_criterion="BPL",
                            stochastic_sampling=False,
                            solving_type="EF-mstage")
        aircondpb.SeedCount = seed
        res.append(aircondpb.run())
        seed = aircondpb.SeedCount
    
    print(res)
        
        