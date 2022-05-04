# special for ph debugging DLW Dec 2018
# unlimited crops
# ALL INDEXES ARE ZERO-BASED
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2018 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# special scalable farmer for stress-testing

import pyomo.environ as pyo
import numpy as np
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils

# Use this random stream:
farmerstream = np.random.RandomState()

def scenario_creator(
    scenario_name, use_integer=False, sense=pyo.minimize, crops_multiplier=1,
    num_scens=None
):
    """ Create a scenario for the (scalable) farmer example.
    
    Args:
        scenario_name (str):
            Name of the scenario to construct.
        use_integer (bool, optional):
            If True, restricts variables to be integer. Default is False.
        sense (int, optional):
            Model sense (minimization or maximization). Must be either
            pyo.minimize or pyo.maximize. Default is pyo.minimize.
        crops_multiplier (int, optional):
            Factor to control scaling. There will be three times this many
            crops. Default is 1.
        num_scens (int, optional):
            Number of scenarios. We use it to compute _mpisppy_probability. 
            Default is None.
    """
    # scenario_name has the form <str><int> e.g. scen12, foobar7
    # The digits are scraped off the right of scenario_name using regex then
    # converted mod 3 into one of the below avg./avg./above avg. scenarios
    scennum   = sputils.extract_num(scenario_name)
    basenames = ['BelowAverageScenario', 'AverageScenario', 'AboveAverageScenario']
    basenum   = scennum  % 3
    groupnum  = scennum // 3
    scenname  = basenames[basenum]+str(groupnum)

    # The RNG is seeded with the scenario number so that it is
    # reproducible when used with multiple threads.
    # NOTE: if you want to do replicates, you will need to pass a seed
    # as a kwarg to scenario_creator then use seed+scennum as the seed argument.
    farmerstream.seed(scennum)

    # Check for minimization vs. maximization
    if sense not in [pyo.minimize, pyo.maximize]:
        raise ValueError("Model sense Not recognized")

    # Create the concrete model object
    model = pysp_instance_creation_callback(
        scenname,
        use_integer=use_integer,
        sense=sense,
        crops_multiplier=crops_multiplier,
    )

    # Create the list of nodes associated with the scenario (for two stage,
    # there is only one node associated with the scenario--leaf nodes are
    # ignored).
    model._mpisppy_node_list = [
        scenario_tree.ScenarioNode(
            name="ROOT",
            cond_prob=1.0,
            stage=1,
            cost_expression=model.FirstStageCost,
            nonant_list=[model.DevotedAcreage],
            scen_model=model,
        )
    ]
    
    #Add the probability of the scenario
    if num_scens is not None :
        model._mpisppy_probability = 1/num_scens
    return model

def pysp_instance_creation_callback(
    scenario_name, use_integer=False, sense=pyo.minimize, crops_multiplier=1
):
    # long function to create the entire model
    # scenario_name is a string (e.g. AboveAverageScenario0)
    #
    # Returns a concrete model for the specified scenario

    # scenarios come in groups of three
    scengroupnum = sputils.extract_num(scenario_name)
    scenario_base_name = scenario_name.rstrip("0123456789")
    
    model = pyo.ConcreteModel()

    def crops_init(m):
        retval = []
        for i in range(crops_multiplier):
            retval.append("WHEAT"+str(i))
            retval.append("CORN"+str(i))
            retval.append("SUGAR_BEETS"+str(i))
        return retval

    model.CROPS = pyo.Set(initialize=crops_init)

    #
    # Parameters
    #

    model.TOTAL_ACREAGE = 500.0 * crops_multiplier

    def _scale_up_data(indict):
        outdict = {}
        for i in range(crops_multiplier):
           for crop in ['WHEAT', 'CORN', 'SUGAR_BEETS']:
               outdict[crop+str(i)] = indict[crop]
        return outdict
        
    model.PriceQuota = _scale_up_data(
        {'WHEAT':100000.0,'CORN':100000.0,'SUGAR_BEETS':6000.0})

    model.SubQuotaSellingPrice = _scale_up_data(
        {'WHEAT':170.0,'CORN':150.0,'SUGAR_BEETS':36.0})

    model.SuperQuotaSellingPrice = _scale_up_data(
        {'WHEAT':0.0,'CORN':0.0,'SUGAR_BEETS':10.0})

    model.CattleFeedRequirement = _scale_up_data(
        {'WHEAT':200.0,'CORN':240.0,'SUGAR_BEETS':0.0})

    model.PurchasePrice = _scale_up_data(
        {'WHEAT':238.0,'CORN':210.0,'SUGAR_BEETS':100000.0})

    model.PlantingCostPerAcre = _scale_up_data(
        {'WHEAT':150.0,'CORN':230.0,'SUGAR_BEETS':260.0})

    #
    # Stochastic Data
    #
    Yield = {}
    Yield['BelowAverageScenario'] = \
        {'WHEAT':2.0,'CORN':2.4,'SUGAR_BEETS':16.0}
    Yield['AverageScenario'] = \
        {'WHEAT':2.5,'CORN':3.0,'SUGAR_BEETS':20.0}
    Yield['AboveAverageScenario'] = \
        {'WHEAT':3.0,'CORN':3.6,'SUGAR_BEETS':24.0}

    def Yield_init(m, cropname):
        # yield as in "crop yield"
        crop_base_name = cropname.rstrip("0123456789")
        if scengroupnum != 0:
            return Yield[scenario_base_name][crop_base_name]+farmerstream.rand()
        else:
            return Yield[scenario_base_name][crop_base_name]

    model.Yield = pyo.Param(model.CROPS,
                            within=pyo.NonNegativeReals,
                            initialize=Yield_init,
                            mutable=True)

    #
    # Variables
    #

    if (use_integer):
        model.DevotedAcreage = pyo.Var(model.CROPS,
                                       within=pyo.NonNegativeIntegers,
                                       bounds=(0.0, model.TOTAL_ACREAGE))
    else:
        model.DevotedAcreage = pyo.Var(model.CROPS, 
                                       bounds=(0.0, model.TOTAL_ACREAGE))

    model.QuantitySubQuotaSold = pyo.Var(model.CROPS, bounds=(0.0, None))
    model.QuantitySuperQuotaSold = pyo.Var(model.CROPS, bounds=(0.0, None))
    model.QuantityPurchased = pyo.Var(model.CROPS, bounds=(0.0, None))

    #
    # Constraints
    #

    def ConstrainTotalAcreage_rule(model):
        return pyo.sum_product(model.DevotedAcreage) <= model.TOTAL_ACREAGE

    model.ConstrainTotalAcreage = pyo.Constraint(rule=ConstrainTotalAcreage_rule)

    def EnforceCattleFeedRequirement_rule(model, i):
        return model.CattleFeedRequirement[i] <= (model.Yield[i] * model.DevotedAcreage[i]) + model.QuantityPurchased[i] - model.QuantitySubQuotaSold[i] - model.QuantitySuperQuotaSold[i]

    model.EnforceCattleFeedRequirement = pyo.Constraint(model.CROPS, rule=EnforceCattleFeedRequirement_rule)

    def LimitAmountSold_rule(model, i):
        return model.QuantitySubQuotaSold[i] + model.QuantitySuperQuotaSold[i] - (model.Yield[i] * model.DevotedAcreage[i]) <= 0.0

    model.LimitAmountSold = pyo.Constraint(model.CROPS, rule=LimitAmountSold_rule)

    def EnforceQuotas_rule(model, i):
        return (0.0, model.QuantitySubQuotaSold[i], model.PriceQuota[i])

    model.EnforceQuotas = pyo.Constraint(model.CROPS, rule=EnforceQuotas_rule)

    # Stage-specific cost computations;

    def ComputeFirstStageCost_rule(model):
        return pyo.sum_product(model.PlantingCostPerAcre, model.DevotedAcreage)
    model.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

    def ComputeSecondStageCost_rule(model):
        expr = pyo.sum_product(model.PurchasePrice, model.QuantityPurchased)
        expr -= pyo.sum_product(model.SubQuotaSellingPrice, model.QuantitySubQuotaSold)
        expr -= pyo.sum_product(model.SuperQuotaSellingPrice, model.QuantitySuperQuotaSold)
        return expr
    model.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

    def total_cost_rule(model):
        if (sense == pyo.minimize):
            return model.FirstStageCost + model.SecondStageCost
        return -model.FirstStageCost - model.SecondStageCost
    model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule, 
                                               sense=sense)

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
    # (only for Amalgamator): add command options unique to farmer
    inparser.add_argument("--crops-multiplier",
                          help="number of crops will be three times this (default 1)",
                          dest="crops_multiplier",
                          type=int,
                          default=1)
    
    inparser.add_argument("--farmer-with-integers",
                          help="make the version that has integers (default False)",
                          dest="use_integer",
                          action="store_true",)
    inparser.set_defaults(use_integer=False)


#=========
def kw_creator(options):
    # (only for Amalgamator): linked to the scenario_creator and inparser_adder
    kwargs = {"use_integer": options.get('use_integer', False),
              "crops_multiplier": options.get('crops_multiplier', 1),
              "num_scens" : options.get('num_scens', None),
              }
    return kwargs


#============================
def scenario_denouement(rank, scenario_name, scenario):
    sname = scenario_name
    s = scenario
    if sname == 'scen0':
        print("Arbitrary sanity checks:")
        print ("SUGAR_BEETS0 for scenario",sname,"is",
               pyo.value(s.DevotedAcreage["SUGAR_BEETS0"]))
        print ("FirstStageCost for scenario",sname,"is", pyo.value(s.FirstStageCost))
        
