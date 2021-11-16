# Copyright 2020, 2021 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Illustrate the use of sequential sampling. Most options are hard-wired.

import sys
import afarmer
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgomator as amalgomator
import mpisppy.confidence_intervals.seqsampling as seqsampling

#============================
def xhat_generator_farmer(scenario_names, solvername="gurobi", solver_options=None,
                          use_integer=False, crops_multiplier=1, start_seed=None):
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
    use_integer: boolean
        indicates the integer farmer version
    crops_multiplier: int
        mulitplied by three to get the total number of crops
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
    
    ama_options = { "EF-2stage": True,
                    "EF_solver_name": solver_name,
                    "EF_solver_options": solver_options,
                    "num_scens": num_scens,
                    "_mpisppy_probability": 1/num_scens,
                    "start_seed":start_seed,
                    }
    #We use from_module to build easily an Amalgomator object
    ama = amalgomator.from_module("afarmer",
                                  ama_options,use_command_line=False)
    #Correcting the building by putting the right scenarios.
    ama.scenario_names = scenario_names
    ama.verbose = False
    ama.run()
    
    # get the xhat
    xhat = sputils.nonant_cache_from_ef(ama.ef)

    return {'ROOT': xhat['ROOT']}
    


def main():
    # use sys.argv to get a few options
    scenario_creator = afarmer.scenario_creator

    crops_multiplier = int(sys.argv[1])
    scen_count = int(sys.argv[2])
    solver_name = sys.argv[3]
    use_integer = False
    
    scenario_creator_kwargs = {
        "use_integer": use_integer,
        "crops_multiplier": crops_multiplier,
    }

    scenario_names = ['Scenario' + str(i) for i in range(scen_count)]

    inneroptions = { "EF_solver_name": solver_name,
                     "start_ups": False,
                     "num_scens": 24,
                     "EF-2stage": True}
    options =  {"num_batches": 5,
                "batch_size": 6,
                "opt":inneroptions}
    scenario_creator_kwargs = afarmer.kw_creator(options)
    options['kwargs'] = scenario_creator_kwargs

    xhat_gen_options = {"scenario_names": scenario_names,
                        "solvername": solver_name,
                        "solver_options": None,
                        "use_integer": use_integer,
                        "crops_multiplier": crops_multiplier,
                        "start_seed": 0,
                        }

    # Bayraksun and Morton
    optionsBM = {'h':1.75,
                 'hprime':0.5, 
                 'eps':0.2, 
                 'epsprime':0.1, 
                 "p":0.1,
                 "q":1.2,
                 "branching_factors": [scen_count],
                 "xhat_gen_options": xhat_gen_options,
                 }

    refmodelname = "afarmer"
    xhat = seqsampling.SeqSampling(refmodelname,
                            xhat_generator_farmer,
                            optionsBM,
                            stochastic_sampling=False,
                            stopping_criterion="BM",
                            )
    return xhat

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("usage python farmer_seqsampling.py {crops_multiplier} {scen_count} {solver_name}")
        print("e.g., python farmer_seqsampling.py 1 3 gurobi")
        quit()
    crops_multiplier = int(sys.argv[1])
    scen_count = int(sys.argv[2])
    solver_name = sys.argv[3]
    
    scenario_creator_kwargs = {
        "use_integer": False,
        "crops_multiplier": crops_multiplier,
    }

    scenario_names = ['Scenario' + str(i) for i in range(scen_count)]
    
    xhat1 = xhat_generator_farmer(scenario_names, solvername=solver_name, solver_options=None,
                                  use_integer=False, crops_multiplier=crops_multiplier,
                                  start_seed=None)
    print(f"test {xhat1 =}")
    xhat = main()
    print(f"final {xhat =}")

