# Copyright 2020, 2021 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Illustrate the use of sequential sampling for programmers.
#
# This is an atypical program in that it allows for a command line
# argument to choose between BM (Bayraksan and Morton) and
# BPL (Bayraksan and Pierre Louis) but provides all
# the command line parameters for both (so either way,
# many command line parameters will be ignored). 

import sys
import numpy as np
import argparse
import farmer
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgamator as amalgamator
import mpisppy.confidence_intervals.seqsampling as seqsampling
import mpisppy.confidence_intervals.confidence_config as confidence_config
from mpisppy.utils import config


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
                    "EF_solver_name": solvername,
                    "EF_solver_options": solver_options,
                    "num_scens": num_scens,
                    "_mpisppy_probability": 1/num_scens,
                    "start_seed":start_seed,
                    }
    #We use from_module to build easily an Amalgamator object
    ama = amalgamator.from_module("farmer",
                                  ama_options,use_command_line=False)
    #Correcting the building by putting the right scenarios.
    ama.scenario_names = scenario_names
    ama.verbose = False
    ama.run()
    
    # get the xhat
    xhat = sputils.nonant_cache_from_ef(ama.ef)

    return {'ROOT': xhat['ROOT']}
    


def main():
    """ Code for farmer sequential sampling (in a function for easier testing)
    Args:
        args (parseargs): the command line arguments object from parseargs
    Returns:
        results (dict): the solution, gap confidence interval and T 
    """
    cfg = config.global_config
    refmodelname = "farmer"
    scenario_creator = farmer.scenario_creator

    scen_count = cfg.num_scens
    solver_name = cfg.EF_solver_name
    crops_multiplier = cfg.crops_multiplier
    
    scenario_names = ['Scenario' + str(i) for i in range(scen_count)]
    
    xhat_gen_options = {"scenario_names": scenario_names,
                        "solvername": solver_name,
                        "solver_options": None,
                        "use_integer": False,
                        "crops_multiplier": crops_multiplier,
                        "start_seed": 0,
                        }

    # simply called "options" by the SeqSampling constructor
    inneroptions = {"solvername": solver_name,
                    "solver_options": None,
                    "sample_size_ratio": cfg.sample_size_ratio,
                    "xhat_gen_options": xhat_gen_options,
                    "ArRP": cfg.ArRP,
                    "kf_xhat": cfg.kf_GS,
                    "kf_xhat": cfg.kf_xhat,
                    "confidence_level": cfg.confidence_level,
                    }

    if cfg.BM_vs_BPL == "BM":
        # Bayraksan and Morton
        optionsBM = {'h': cfg.BM_h,
                     'hprime': cfg.BM_hprime, 
                     'eps': cfg.BM_eps, 
                     'epsprime': cfg.BM_eps_prime, 
                     "p": cfg.BM_p,
                     "q": cfg.BM_q,
                     "xhat_gen_options": xhat_gen_options,
                     }

        optionsBM.update(inneroptions)

        sampler = seqsampling.SeqSampling(refmodelname,
                                xhat_generator_farmer,
                                optionsBM,
                                stochastic_sampling=False,
                                stopping_criterion="BM",
                                solving_type="EF-2stage",
                                )
    else:  # must be BPL
        optionsBPL = {'eps': cfg.BPL_eps, 
                      "c0": cfg.BPL_c0,
                      "n0min": cfg.BPL_n0min,
                      "xhat_gen_options": xhat_gen_options,
                      }

        optionsBPL.update(inneroptions)
        
        ss = int(cfg.BPL_n0min) != 0
        sampler = seqsampling.SeqSampling(refmodelname,
                                xhat_generator_farmer,
                                optionsBPL,
                                stochastic_sampling=ss,
                                stopping_criterion="BPL",
                                solving_type="EF-2stage",
                                )
        
    xhat = sampler.run()
    return xhat

def _parse_args():
    farmer.inparser_adder()
    config.EF2()
    confidence_config.confidence_config()
    confidence_config.sequential_config()
    confidence_config.BM_config()
    confidence_config.BPL_config()  # --help will show both BM and BPL

    config.add_to_config("BM_vs_BPL",
                        description="BM or BPL for Bayraksan and Morton or B and Pierre Louis",
                        domain=str,
                        default=None)
    config.add_to_config("crops_mult",
                         description="There will be 3x this many crops (default 1)",
                         domain=int,
                         default=1)                
    config.add_to_config("xhat1_file",
                        description="File to which xhat1 should be (e.g. to process with zhat4hat.py)",
                        domain=str,
                        default=None)
    # note that num_scens is special until Pyomo config supports positionals
    config.add_to_config("num_scens",
                         description="Number of Scenarios (required, positional)",
                         domain=int,
                         default=-1,
                         argparse=False)   # special

    parser = config.create_parser("farmer_seqsampling")
    # more special treatment of num_scens
    parser.add_argument(
        "num_scens", help="Number of scenarios", type=int
    )
    
    args = parser.parse_args()  # from the command line
    args = config.global_config.import_argparse(args)

    # final special treatment of num_scens
    config.global_config.num_scens = args.num_scens
    cfg = config.global_config

    if cfg.BM_vs_BPL is None:
        raise RuntimeError("--BM-vs_BPL must be given.")
    if cfg.BM_vs_BPL != "BM" and cfg.BM_vs_BPL != "BPL":
        raise RuntimeError(f"--BM-vs_BPL must be BM or BPL (you gave {cfg.BM_vs_BMPL})")
    
    return args



if __name__ == '__main__':

    _parse_args()
    cfg = config.global_config
    
    results = main()
    print(f"Final gap confidence interval results:", results)

    if cfg.xhat1_file is not None:
        print(f"Writing xhat1 to {cfg.xhat1_file}.npy")
        root_nonants =np.fromiter((v for v in results["Candidate_solution"]["ROOT"]), float)
        np.save(cfg.xhat1_file, root_nonants)
