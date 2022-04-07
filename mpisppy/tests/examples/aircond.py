#ReferenceModel for full set of scenarios for AirCond; June 2021
# Dec 2021; numerous enhancements by DLW; do not change defaults
# Feb 2022: Changed all inventory cost coefficients to be positive numbers
# exccept Last Inventory cost, which should be negative.
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
# Do not edit these defaults!
parms = {"mudev": (float, 0.),
         "sigmadev": (float, 40.),
         "start_ups": (bool, False),
         "StartUpCost": (float, 300.),
         "start_seed": (int, 1134),
         "min_d": (float, 0.),
         "max_d": (float, 400.),
         "starting_d": (float, 200.),
         "BeginInventory": (float, 200.),
         "InventoryCost": (float, 0.5),
         "LastInventoryCost": (float, -0.8),
         "Capacity": (float, 200.),
         "RegularProdCost": (float, 1.),
         "OvertimeProdCost": (float, 3.),
         "NegInventoryCost": (float, 5.),
         "QuadShortCoeff": (float, 0)
}

def _demands_creator(sname, sample_branching_factors, root_name="ROOT", **kwargs):

    if "start_seed" not in kwargs:
        raise RuntimeError(f"start_seed not in kwargs={kwargs}")
    start_seed = kwargs["start_seed"]
    max_d = kwargs.get("max_d", 400)
    min_d = kwargs.get("min_d", 0)
    mudev = kwargs.get("mudev", None)
    sigmadev = kwargs.get("sigmadev", None)

    scennum   = sputils.extract_num(sname)
    # Find the right path and the associated seeds (one for each node) using scennum
    prod = np.prod(sample_branching_factors)
    s = int(scennum % prod)
    d = kwargs.get("starting_d", 200)
    demands = [d]
    nodenames = [root_name]
    for bf in sample_branching_factors:
        assert prod%bf == 0
        prod = prod//bf
        nodenames.append(str(s//prod))
        s = s%prod
    
    stagelist = [int(x) for x in nodenames[1:]]
    for t in range(1,len(nodenames)):
        aircondstream.seed(start_seed+sputils.node_idx(stagelist[:t],sample_branching_factors))
        d = min(max_d,max(min_d,d+aircondstream.normal(mudev,sigmadev)))
        demands.append(d)
    
    return demands,nodenames

def general_rho_setter(scenario_instance, rho_scale_factor=1.0):

    computed_rhos = []

    for t in scenario_instance.T[:-1]:
        computed_rhos.append((id(scenario_instance.stage_models[t].RegularProd),
                              pyo.value(scenario_instance.stage_models[t].RegularProdCost) * rho_scale_factor))
        computed_rhos.append((id(scenario_instance.stage_models[t].OvertimeProd),
                              pyo.value(scenario_instance.stage_models[t].OvertimeProdCost) * rho_scale_factor))

    return computed_rhos

def dual_rho_setter(scenario_instance):
    return general_rho_setter(scenario_instance, rho_scale_factor=0.0001)

def primal_rho_setter(scenario_instance):
    return general_rho_setter(scenario_instance, rho_scale_factor=0.01)


def _StageModel_creator(time, demand, last_stage, **kwargs):
    # create a single stage of an aircond model; do not change defaults (Dec 2021)

    def _kw(pname):
        return kwargs.get(pname, parms[pname][1])
    
    model = pyo.ConcreteModel()
    model.T = [time]

    #Parameters
    #Demand
    model.Demand = demand
    #Inventory Cost: model.InventoryCost = -0.8 if last_stage else 0.5
    model.InventoryCost = _kw("InventoryCost")
    model.LastInventoryCost = _kw("LastInventoryCost")
    #Regular Capacity
    model.Capacity = _kw("Capacity")
    #Regular Production Cost
    model.RegularProdCost = _kw("RegularProdCost")
    #Overtime Production Cost
    model.OvertimeProdCost = _kw("OvertimeProdCost")

    model.max_T = 25 # this is for the start-up cost constraints, 
    model.bigM = model.Capacity * model.max_T
    
    model.start_ups = _kw("start_ups")

    if model.start_ups:
        # Start-up Cost
        model.StartUpCost = _kw("StartUpCost")

    # Negative Inventory Cost
    model.NegInventoryCost = _kw("NegInventoryCost")
    model.QuadShortCoeff = _kw("QuadShortCoeff")
    
    #Variables
    model.RegularProd = pyo.Var(domain=pyo.NonNegativeReals,
                                bounds = (0, model.bigM))
    
    model.OvertimeProd = pyo.Var(domain=pyo.NonNegativeReals,
                                bounds = (0, model.bigM))

    model.Inventory = pyo.Var(domain=pyo.Reals,
                              bounds = (-model.bigM, model.bigM))

    if model.start_ups:
        # start-up cost variable
        model.StartUp = pyo.Var(within=pyo.Binary)

    #Constraints (and some auxilliary variables)
    def CapacityRule(m):
        return m.RegularProd<=m.Capacity
    model.MaximumCapacity = pyo.Constraint(rule=CapacityRule)

    if model.start_ups:
        model.RegStartUpConstraint = pyo.Constraint(
            expr=model.bigM * model.StartUp >= model.RegularProd + model.OvertimeProd)

    # grab positive and negative parts of inventory
    assert model.InventoryCost > 0, model.InventoryCost
    assert model.NegInventoryCost > 0, model.NegInventoryCost
    model.negInventory = pyo.Var(domain=pyo.NonNegativeReals, initialize=0.0, bounds = (0, model.bigM))
    model.posInventory = pyo.Var(domain=pyo.NonNegativeReals, initialize=0.0, bounds = (0, model.bigM))
    model.doleInventory = pyo.Constraint(expr=model.Inventory == model.posInventory - model.negInventory)

    # create the inventory cost expression
    if not last_stage:
        model.LinInvenCostExpr = pyo.Expression(expr = model.InventoryCost*model.posInventory\
                                                + model.NegInventoryCost*model.negInventory)
    else:  
        assert model.LastInventoryCost < 0, f"last stage inven cost: {model.LastInventoryCost}"        
        model.LinInvenCostExpr = pyo.Expression(expr = model.LastInventoryCost*model.posInventory\
                                                + model.NegInventoryCost*model.negInventory)

    if model.QuadShortCoeff > 0 and not last_stage:
        model.InvenCostExpr =  pyo.Expression(expr = model.LinInvenCostExpr +\
                                              model.QuadShortCoeff * model.negInventory * model.negInventory)
    else:
        assert model.QuadShortCoeff >= 0, model.QuadShortCoeff
        model.InvenCostExpr =  pyo.Expression(expr = model.LinInvenCostExpr)
    
    #Objective
    def stage_objective(m):
        expr =  m.RegularProdCost*m.RegularProd +\
                m.OvertimeProdCost*m.OvertimeProd +\
                m.InvenCostExpr

        if m.start_ups:
            expr += m.StartUpCost * m.StartUp 
        return expr

    model.StageObjective = pyo.Objective(rule=stage_objective,sense=pyo.minimize)
    
    return model

#Assume that demands has been drawn before
def aircond_model_creator(demands, **kwargs):
    # create a single aircond model for the given demands
    # branching_factors=None, num_scens=None, mudev=0, sigmadev=40, start_seed=0, start_ups=None):
    # typing aids...
    start_seed = kwargs["start_seed"]

    start_ups = kwargs.get("start_ups", parms["start_ups"][1])
    
    model = pyo.ConcreteModel()
    num_stages = len(demands)
    model.T = range(1,num_stages+1) #Stages 1,2,...,T
    model.max_T = 25
    if model.T[-1] > model.max_T:
        raise RuntimeError(
            f'The number of stages exceeds {model.max_T}')
    
    model.start_ups = start_ups

    #Parameters
    model.BeginInventory = kwargs.get("BeginInventory", parms["BeginInventory"][1])
    
    #Creating stage models
    model.stage_models = {}
    for t in model.T:
        last_stage = (t==num_stages)
        model.stage_models[t] = _StageModel_creator(t, demands[t-1],
                                                    last_stage=last_stage, **kwargs)

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
        expr = stage_m.RegularProdCost*stage_m.RegularProd +\
               stage_m.OvertimeProdCost*stage_m.OvertimeProd +\
               stage_m.InvenCostExpr
        if stage_m.start_ups:
            expr += stage_m.StartUpCost * stage_m.StartUp
        return expr
    
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
        
        nonant_ef_suppl_list = [model.stage_models[stage].Inventory]
        if model.start_ups:
            nonant_ef_suppl_list.append(model.stage_models[stage].StartUp)

        if stage ==1:
            ndn="ROOT"
            TreeNodes.append(scenario_tree.ScenarioNode(name=ndn,
                                                        cond_prob=1.0,
                                                        stage=stage,
                                                        cost_expression=model.stage_models[stage].StageObjective,
                                                        nonant_list=nonant_list,
                                                        scen_model=model,
                                                        nonant_ef_suppl_list = nonant_ef_suppl_list
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
                                                        nonant_ef_suppl_list = nonant_ef_suppl_list,
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
                                                        nonant_ef_suppl_list = nonant_ef_suppl_list,
                                                        parent_name = parent_ndn
                                                       )
                             )
    return(TreeNodes)

        
def scenario_creator(sname, **kwargs):

    start_seed = kwargs['start_seed']
    if "branching_factors" not in kwargs:
        raise RuntimeError("scenario_creator for aircond needs branching_factors in kwargs")
    branching_factors = kwargs["branching_factors"]

    demands,nodenames = _demands_creator(sname, branching_factors, root_name="ROOT", **kwargs)
    
    model = aircond_model_creator(demands, **kwargs)
    
    
    #Constructing the nodes used by the scenario
    model._mpisppy_node_list = MakeNodesforScen(model, nodenames, branching_factors)
    model._mpisppy_probability = 1 / np.prod(branching_factors)
    """
    from mpisppy import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        with open("efmodel.txt", "w") as fileh:
            model.pprint(fileh)
    quit()
    """
    return(model)


def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    """ Create a scenario within a sample tree. Mainly for multi-stage and simple for two-stage.
    Args:
        sname (string): scenario name to be created
        stage (int >=1 ): for stages > 1, fix data based on sname in earlier stages
        sample_branching_factors (list of ints): branching factors for the sample tree
        seed (int): To allow random sampling (for some problems, it might be scenario offset)
        given_scenario (Pyomo concrete model): if not None, use this to get data for ealier stages
        scenario_creator_kwargs (dict): keyword args for the standard scenario creator funcion
    Returns:
        scenario (Pyomo concrete model): A scenario for sname with data in stages < stage determined
                                         by the arguments
    """
    
    # Finding demands from stage 1 to t
    starting_d = scenario_creator_kwargs.get("starting_d", parms["starting_d"][1])
    if given_scenario is None:
        if stage == 1:
            past_demands = [starting_d]
        else:
            raise RuntimeError(f"sample_tree_scen_creator for aircond needs a 'given_scenario' argument if the starting stage is greater than 1")
    else:
        past_demands = [given_scenario.stage_models[t].Demand for t in given_scenario.T if t<=stage]

    kwargs = scenario_creator_kwargs.copy()
    kwargs["start_seed"] = seed
        
    #Finding demands for stages after t (note the dynamic seed)
    future_demands,nodenames = _demands_creator(sname, sample_branching_factors, 
                                                root_name='ROOT'+'_0'*(stage-1), **kwargs)
    
    demands = past_demands+future_demands[1:] #The demand at the starting stage is in both past and future demands
    
    model = aircond_model_creator(demands, **scenario_creator_kwargs)
    
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
    # Do not change the defaults.
    def _doone(name, helptext, argname=None):
        # The name should be the name in parms
        # helptext should not include the default
        aname = name.replace("_", "-") if argname is None else argname
        h = f"{helptext} (default {parms[name][1]})"
        inparser.add_argument(f"--{aname}",
                              help=h,
                              dest=name,
                              type=parms[name][0],
                              default=parms[name][1])


    _doone("mudev", "average deviation of demand between two periods", argname="mu-dev")
    _doone("sigmadev", "standard deviation of deviation of demand between two periods", argname="sigma-dev")

    d = parms["start_ups"][1]
    inparser.add_argument("--start-ups",
                          help="Include start-up costs in model, resulting in a MPI (default {d})",
                          dest="start_ups",
                          action="store_true"
                          )
    inparser.set_defaults(start_ups=d)
    
    _doone("StartUpCost", helptext="Cost if production in a period is non-zero and start-up is True")
    _doone("start_seed", helptext="random number seed")
    _doone("min_d", helptext="minimum demand in a period")
    _doone("max_d", helptext="maximum demand in a period")
    _doone("starting_d", helptext="period 0 demand")
    _doone("InventoryCost", helptext="Inventory cost per period per item")
    _doone("BeginInventory", helptext="Inital Inventory")
    _doone("LastInventoryCost", helptext="Inventory `cost` (should be negative) in last period)")
    _doone("Capacity", helptext="Per period regular time capacity")
    _doone("RegularProdCost", helptext="Regular time producion cost")
    _doone("OvertimeProdCost", helptext="Overtime (or subcontractor) production cost")
    _doone("NegInventoryCost", helptext="Linear coefficient for backorders (should be negative; not used in last stage)")
    _doone("QuadShortCoeff", helptext="Coefficient for backorders squared (should be nonnegative; not used in last stage)")
    
    return inparser


#=========
def kw_creator(options):
    # TBD: re-write this function...
    if "kwargs" in options:
        return options["kwargs"]
    
    kwargs = dict()
    
    def _kwarg(option_name, default = None, arg_name=None):
        # options trumps args
        retval = options.get(option_name)
        if retval is not None:
            kwargs[option_name] = retval
            return 
        args = options.get('args')
        aname = option_name if arg_name is None else arg_name
        retval = getattr(args, aname) if hasattr(args, aname) else None
        retval = default if retval is None else retval
        kwargs[option_name] = retval
            
    # (only for Amalgamator): linked to the scenario_creator and inparser_adder
    # for confidence intervals, we need to see if the values are in args
    _kwarg("branching_factors")
    for idx, tpl in parms.items():
        _kwarg(idx, tpl[1])

    if kwargs["start_ups"] is None:
        raise ValueError(f"kw_creator called, but no value given for start_ups, options={options}")
    if kwargs["start_seed"] is None:
        raise ValueError(f"kw_creator called, but no value given for start_seed, options={options}")

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
    ama = amalgamator.from_module("mpisppy.tests.examples.aircond",
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
    refmodel = "mpisppy.tests.examples.aircond" # WARNING: Change this in SPInstances
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
   
        
        
