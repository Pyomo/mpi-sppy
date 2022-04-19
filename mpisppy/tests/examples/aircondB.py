# This software is distributed under the 3-clause BSD License.
#ReferenceModel for full set of scenarios for AirCond; June 2021
# PICKLE BUNDLE VERSION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Dec 2021; numerous enhancements by DLW; do not change defaults
# Feb 2022: Changed all inventory cost coefficients to be positive numbers
# exccept Last Inventory cost, which should be negative.
import os
import numpy as np
import time
import pyomo.environ as pyo
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgamator as amalgamator
import mpisppy.utils.pickle_bundle as pickle_bundle
import mpisppy.tests.examples.aircond as base_aircond
from mpisppy.utils.sputils import attach_root_node
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
    # create replicable pseudo random demand steps
    # DLW on 3March2022: The seed and therefore the demand (directly) are determined by
    # the node index so the pickle bundler needs to mess with the start_seed that is passed in here...
    # (this is probably the same as the base_aircond version and probably should be).
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
    """ TBD: make this work with proper bundles, which are two stage problems when solved"""    
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
    return base_aircond._StageModel_creator(time, demand, last_stage, kwargs)


#Assume that demands has been drawn before
def aircond_model_creator(demands, **kwargs):
    return base_aircond.aircond_model_creator(demands, **kwargs)

def MakeNodesforScen(model,nodenames,branching_factors,starting_stage=1):
    return base_aircond.MakeNodesforScen(model,nodenames,branching_factors,starting_stage=starting_stage)

        
def scenario_creator(sname, **kwargs):
    """
    NOTE: modified Feb/March 2022 to be able to return a bundle if the name
    is Bundle_firstnum_lastnum (e.g. Bundle_14_28)
    This returns a Pyomo model (either for scenario, or the EF of a bundle)
    """
    def _bunBFs(branching_factors, bunsize):
        # return branching factors for the bundle's EF
        assert len(branching_factors) > 1
        beyond2size = np.prod(branching_factors[1:])
        if bunsize % beyond2size!= 0:
            raise RuntimeError(f"Bundles must consume entire second stage nodes {beyond2size} {bunsize}")
        bunBFs = [bunsize // beyond2size] + branching_factors[1:]  # branching factors in the bundle
        return bunBFs

    
    # NOTE: start seed is not used here (noted Feb 2022)
    start_seed = kwargs['start_seed']
    if "branching_factors" not in kwargs:
        raise RuntimeError("scenario_creator for aircond needs branching_factors in kwargs")
    branching_factors = kwargs["branching_factors"]

    if "scen" in sname:

        demands,nodenames = _demands_creator(sname, branching_factors, root_name="ROOT", **kwargs)

        model = aircond_model_creator(demands, **kwargs)

        #Constructing the nodes used by the scenario
        model._mpisppy_node_list = MakeNodesforScen(model, nodenames, branching_factors)
        model._mpisppy_probability = 1 / np.prod(branching_factors)

        return model 

    elif "Bundle" in sname:
        firstnum = int(sname.split("_")[1])
        lastnum = int(sname.split("_")[2])
        snames = [f"scen{i}" for i in range(firstnum, lastnum+1)]

        if kwargs.get("unpickle_bundles_dir") is not None:
            fname = os.path.join(kwargs["unpickle_bundles_dir"], sname+".pkl")
            bundle =  pickle_bundle.dill_unpickle(fname)
            return bundle

        # if we are still here, we have to create the bundle (we did not load it)
        bunkwargs = kwargs.copy()
        bunkwargs["branching_factors"] = _bunBFs(branching_factors, len(snames))
        # The next line is needed, but eliminates comparison of pickle and non-pickle
        bunkwargs["start_seed"] = start_seed+firstnum * len(branching_factors) # the usual block of seeds
        bundle = sputils.create_EF(snames, scenario_creator,
                                   scenario_creator_kwargs=bunkwargs, EF_name=sname,
                                   nonant_for_fixed_vars = False)
        # It simplifies things if we assume that the bundles consume entire second stage nodes,
        # then all we need is a root node and the only nonants that need to be reported are
        # at the root node (otherwise, more coding is required here to figure out which nodes and Vars
        # are shared with other bundles)
        bunsize = (lastnum-firstnum+1)
        N = np.prod(branching_factors)
        numbuns = N / bunsize
        nonantlist = [v for idx,v in bundle.ref_vars.items() if idx[0] =="ROOT"]
        attach_root_node(bundle, 0, nonantlist)
        # scenarios are equally likely so bundles are too
        bundle._mpisppy_probability = 1/numbuns
        if kwargs.get("pickle_bundles_dir") is not None:
            # note that sname is a bundle name
            fname = os.path.join(kwargs["pickle_bundles_dir"], sname+".pkl")
            pickle_bundle.dill_pickle(bundle, fname)
        return bundle
    else:
        raise RuntimeError (f"Scenario name does not have scen or Bundle: {sname}")
        

def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    return base_aircond.sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=given_scenario, **scenario_creator_kwargs)
                

#=========
def scenario_names_creator(num_scens,start=None):
    # (only for Amalgamator): return the full list of num_scens scenario names
    # if start!=None, the list starts with the 'start' labeled scenario
    if (start is None) :
        start=0
    return [f"scen{i}" for i in range(start,start+num_scens)]
        

#=========
def inparser_adder(inparser):
    inparser = base_aircond.inparser_adder(inparser)
    # special "proper" bundle arguments
    inparser = pickle_bundle.pickle_bundle_parser(inparser)
    
    return inparser


#=========
def kw_creator(options):
    kwargs = base_aircond.kw_creator(options)

    # proper bundle args are special
    args = options.get("args")
    kwargs["pickle_bundles_dir"] = args.pickle_bundles_dir
    kwargs["unpickle_bundles_dir"] = args.unpickle_bundles_dir

    return kwargs


#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass


#============================
def xhat_generator_aircond(scenario_names, solvername="gurobi", solver_options=None,
                           branching_factors=None, mudev = 0, sigmadev = 40,
                           start_ups=None, start_seed = 0):
    return base_aircond.xhat_generator_aircond(scenario_names, solvername="gurobi", solver_options=solver_options,
                                               branching_factors=branching_factors, mudev = mudev, sigmadev = sigmadev,
                                               start_ups=start_ups, start_seed = start_seed)


if __name__ == "__main__":
    print("not directly runnable.")
   
        
        
