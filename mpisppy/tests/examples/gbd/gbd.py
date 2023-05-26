#ReferenceModel for full set of scenarios for GBD; Aug 2021
#From original Ferguson and Dantzig 1956
#Extended scenarios as done in Seq Sampling by B&M

import pyomo.environ as pyo
import numpy as np
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgamator as amalgamator
import json

# Use this random stream:
gbdstream= np.random.RandomState()

with open('gbd_data/gbd_extended_data.json','r') as openfile:
   gbd_extended_data = json.load(openfile)

def GBD_model_creator(seed):

    gbdstream.seed(seed)
    random_array = gbdstream.rand(5) #We only use 5 random numbers
    
    #
    # Model
    #
    
    model = pyo.ConcreteModel()
    model.aircraftType = range(4)
    model.route = range(5)

    # number of aircraft type i sent on route j
    model.x = pyo.Var(model.aircraftType, model.route, within=pyo.NonNegativeReals)
    # these aircrafts cannot fly these routes:
    model.x[1,0].fix(0)
    model.x[2,0].fix(0)
    model.x[2,2].fix(0)

    # excess aircraft
    model.aircraftSlack = pyo.Var(model.aircraftType, within=pyo.NonNegativeReals)

    # excess / deficit demand in hundreds of passengers
    model.passengerSlack_pos = pyo.Var(model.route, within=pyo.NonNegativeReals)
    model.passengerSlack_neg = pyo.Var(model.route, within=pyo.NonNegativeReals)
    

    #
    # Parameters
    #

    model.numAircrafts = [10,19,25,15] # fixed aircraft inventory

    # (hundreds) of passengers which can be hauled per month for
    # each aircraft type and route (see paper for formulation)
    model.p = {
        (1,1): 16, (1,2):15, (1,3):28, (1,4):23, (1,5):81, (1,6):0,

        (2,1):0, (2,2): 10, (2,3):14, (2,4):15, (2,5):57, (2,6):0,

        (3,1):0, (3,2):5, (3,3):0, (3,4):7, (3,5):29, (3,6):0,

        (4,1):9, (4,2):11, (4,3):22, (4,4):17, (4,5):55, (4,6):0,

        (5,1):1, (5,2):1, (5,3):1, (5,4):1, (5,5):1, (5,6):0
    }

    # cost per month (in thousands) for each aircraft type for each route
    model.c = {
        (1,1): 18, (1,2):21, (1,3):18, (1,4):16, (1,5):10, (1,6):0,

        (2,1):0, (2,2): 15, (2,3):16, (2,4):14, (2,5):9, (2,6):0,

        (3,1):0, (3,2):10, (3,3):0, (3,4):9, (3,5):6, (3,6):0,

        (4,1):17, (4,2):16, (4,3):17, (4,4):15, (4,5):10, (4,6):0,

        (5,1):13, (5,2):13, (5,3):7, (5,4):7, (5,5):1, (5,6):0
    }

    #demand uncertainty

    possible_demands = (gbd_extended_data['r1_dmds'], 
                        gbd_extended_data['r2_dmds'],
                        gbd_extended_data['r3_dmds'],
                        gbd_extended_data['r4_dmds'],
                        gbd_extended_data['r5_dmds']
                       )
    demand_probs = (gbd_extended_data['r1_prbs'], 
                        gbd_extended_data['r2_prbs'],
                        gbd_extended_data['r3_prbs'],
                        gbd_extended_data['r4_prbs'],
                        gbd_extended_data['r5_prbs']
                       )

    demand_cumprobs=(np.flip(np.cumsum(np.flip(demand_probs[0]))),
                    np.flip(np.cumsum(np.flip(demand_probs[1]))),
                    np.flip(np.cumsum(np.flip(demand_probs[2]))),
                    np.flip(np.cumsum(np.flip(demand_probs[3]))),
                    np.flip(np.cumsum(np.flip(demand_probs[4]))))

    # original demands fron 1956 paper:
    # possible_demands = ([20, 22, 25, 27, 30], 
    #                 [5, 15],
    #                 [14, 16, 18, 20, 22],
    #                 [1, 5, 8, 10, 34],
    #                 [58, 60, 62]
    #                )
    # demand_probs = ([.2, .05, .35, .2, .2], 
    #                 [.3, .7],
    #                 [.1, .2, .4, .2, .1],
    #                 [.2, .2, .3, .2, .1],
    #                 [.1, .8, .1]
    #                     )
    # demand_cumprobs = ([1, .8, .75, .4, .2],
    #                    [1, .7],
    #                    [1, .9, .7, .3, .1],
    #                    [1, .8, .6, .3, .1],
    #                    [1, .9, .1]
    #                     )


    # Assign demands

    def demands_init(m,g):
        rd = random_array[g]
        j = np.searchsorted(np.flip(demand_cumprobs[g]),rd)
        return possible_demands[g][len(demand_cumprobs[g])-1 - j]

    model.passengerDemand = pyo.Param(model.route, within=pyo.NonNegativeReals,
                               initialize=demands_init)

    #
    # Constraints
    #
    
    # aircraft inventory constraint
    def AircraftSupply_rule(m, a):
        return sum(model.x[a,j] for j in model.route) + model.aircraftSlack[a] == model.numAircrafts[a]

    model.SatisfyAircraftConstraints = pyo.Constraint(model.aircraftType, rule=AircraftSupply_rule)

    # route demand constraint
    def RouteDemand_rule(m,r):
        return sum(model.p[i+1,r+1]*model.x[i,r] for i in model.aircraftType) +\
            model.p[5,r+1]*(model.passengerSlack_pos[r] - model.passengerSlack_neg[r]) == model.passengerDemand[r]

    model.SatisftyDemandConstraints = pyo.Constraint(model.route, rule=RouteDemand_rule)

    #
    # Objective (thousands of dollars)
    #

    def ComputeOperatingCosts(m):
        return sum(model.c[i+1,j+1]*model.x[i,j] for i in model.aircraftType for j in model.route) 

    model.OperatingCosts = pyo.Expression(rule=ComputeOperatingCosts)

    def ComputePassengerRevenueLost(m):
        return sum(model.c[5, j+1]*model.passengerSlack_pos[j] for j in model.route)

    model.PassengerRevenueLost = pyo.Expression(rule=ComputePassengerRevenueLost)

    def ComputeTotalCost(m):
        return model.OperatingCosts + model.PassengerRevenueLost

    model.obj = pyo.Objective(rule=ComputeTotalCost)

    return(model)


def scenario_creator(sname, num_scens=None):
    scennum   = sputils.extract_num(sname)
    model = GBD_model_creator(scennum)
    
    # Create the list of nodes associated with the scenario (for two stage,
    # there is only one node associated with the scenario--leaf nodes are
    # ignored).
    model._mpisppy_node_list = [
        scenario_tree.ScenarioNode(
            name="ROOT",
            cond_prob=1.0,
            stage=1,
            cost_expression=model.obj,
            nonant_list=[model.x],
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
    # (only for Amalgamator): add command options unique to gbd
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
def xhat_generator_gbd(scenario_names, solver_name="gurobi", solver_options=None):
    '''
    For sequential sampling.
    Takes scenario names as input and provide the best solution for the 
        approximate problem associated with the scenarios.
    Parameters
    ----------
    scenario_names: int
        Names of the scenario we use
    solver_name: str, optional
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
                    "EF_solver_name": solver_name,
                    "EF_solver_options": solver_options,
                    "num_scens": num_scens,
                    "_mpisppy_probability": 1/num_scens,
                    }
    #We use from_module to build easily an Amalgamator object
    ama = amalgamator.from_module("mpisppy.tests.examples.gbd.gbd",
                                  ama_options,use_command_line=False)
    #Correcting the building by putting the right scenarios.
    ama.scenario_names = scenario_names
    ama.run()
    
    # get the xhat
    xhat = sputils.nonant_cache_from_ef(ama.ef)

    return xhat

if __name__ == "__main__":

    num_scens = 300
    solver_name = 'gurobi_direct'
    solver_options=None
    
    scenario_names = scenario_names_creator(num_scens,start=6000)

    ama_options = { "EF-2stage": True,
                    "EF_solver_name": solver_name,
                    "EF_solver_options": solver_options,
                    "num_scens": num_scens,
                    "_mpisppy_probability": None,
                    }
    #We use from_module to build easily an Amalgamator object
    ama = amalgamator.from_module("mpisppy.tests.examples.gbd.gbd",
                                  ama_options,use_command_line=False)
    #Correcting the building by putting the right scenarios.
    ama.scenario_names = scenario_names
    ama.run()
    print(f"inner bound=", ama.best_inner_bound)
    print(f"outer bound=", ama.best_outer_bound)

    xhat = sputils.nonant_cache_from_ef(ama.ef)
    print("xhat=",xhat['ROOT'])

    from mpisppy.confidence_intervals.seqsampling import SeqSampling
    optionsFSP = {'eps': 8.0,
                  'solver_name': "gurobi_direct",
                  "c0":300,
                  "kf_Gs":25,
                  "kf_xhat":25}

    #from seq samp paper:
    optionsBM_GBD_300 =  { 'h':0.18,
                    'hprime':0.015, 
                    'eps':2e-7, 
                    'epsprime':1e-7, 
                    "p":0.191,
                    "kf_Gs":25,
                    "kf_xhat":1,
                    "solver_name":"gurobi_direct",
                    }
    optionsBM_GBD_500 =  { 'h':0.1427,
                    'hprime':0.015, 
                    'eps':2e-7, 
                    'epsprime':1e-7, 
                    "p":0.191,
                    "kf_Gs":25,
                    "kf_xhat":1,
                    "solver_name":"gurobi_direct",
                    }

    gbd_pb = SeqSampling("mpisppy.tests.examples.gbd.gbd",
                            xhat_generator_gbd, 
                            optionsFSP,
                            stopping_criterion="BPL",
                            stochastic_sampling=False)

    res = gbd_pb.run()
    print(res)
