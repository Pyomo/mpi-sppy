# Copyright 2020, 2021 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Employ the aircond model to illustrate how to use sequential sampling.
#

import sys
import numpy as np
import argparse
import mpisppy.tests.examples.aircond as aircond
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgamator as amalgamator
import mpisppy.confidence_intervals.multi_seqsampling as multi_seqsampling
import mpisppy.confidence_intervals.confidence_parsers as confidence_parsers
from mpisppy.utils import baseparsers

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

    return {'ROOT': xhat['ROOT']}



def main(args):
    """ Code for aircond sequential sampling (in a function for easier testing)
    Args:
        args (parseargs): the command line arguments object from parseargs
    Returns:
        results (dict): the solution, gap confidence interval and T 
    """
    refmodelname = "mpisppy.tests.examples.aircond"
    scenario_creator = aircond.scenario_creator

    BFs = args.branching_factors
    num_scens = np.prod(BFs)
    solver_name = args.solver_name
    mudev = args.mudev
    sigmadev = args.sigmadev
    scenario_creator_kwargs = {"num_scens" : num_scens,
                               "branching_factors": BFs,
                               "mudev": mudev,
                               "sigmadev": sigmadev,
                               "start_ups": False,
                               "start_seed": args.seed,
                               }
    
    scenario_names = ['Scenario' + str(i) for i in range(num_scens)]
    
    xhat_gen_options = {"scenario_names": scenario_names,
                        "solvername": solver_name,
                        "solver_options": None,
                        "branching_factors" : BFs,
                        "mudev": mudev,
                        "sigmadev": sigmadev,
                        "start_ups": False,
                        "start_seed": args.seed,
                        }

    # simply called "options" by the SeqSampling constructor
    inneroptions = {"solvername": solver_name,
                    "branching_factors": BFs,
                    "solver_options": None,
                    "sample_size_ratio": args.sample_size_ratio,
                    "xhat_gen_options": xhat_gen_options,
                    "ArRP": args.ArRP,
                    "kf_xhat": args.kf_GS,
                    "kf_xhat": args.kf_xhat,
                    "confidence_level": args.confidence_level,
                    "start_ups": False,
                    }

    if args.BM_vs_BPL == "BM":
        # Bayraksan and Morton
        optionsBM = {'h': args.BM_h,
                     'hprime': args.BM_hprime, 
                     'eps': args.BM_eps, 
                     'epsprime': args.BM_eps_prime, 
                     "p": args.BM_p,
                     "q": args.BM_q,
                     "xhat_gen_options": xhat_gen_options,
                     }

        optionsBM.update(inneroptions)

        sampler = multi_seqsampling.IndepScens_SeqSampling(refmodelname,
                                          xhat_generator_aircond,
                                          optionsBM,
                                          stochastic_sampling=False,
                                          stopping_criterion="BM",
                                          solving_type="EF-mstage",
                                          )
    else:  # must be BPL
        optionsBPL = {'eps': args.BPL_eps, 
                      "c0": args.BPL_c0,
                      "n0min": args.BPL_n0min,
                      "xhat_gen_options": xhat_gen_options,
                      }

        optionsBPL.update(inneroptions)
        
        ss = int(args.BPL_n0min) != 0
        sampler = multi_seqsampling.IndepScens_SeqSampling(refmodelname,
                                xhat_generator_aircond,
                                optionsBPL,
                                stochastic_sampling=ss,
                                stopping_criterion="BPL",
                                solving_type="EF-mstage",
                                )
        
    xhat = sampler.run()
    return xhat

def _parse_args():
    parser = baseparsers._basic_multistage("aircond_seqsampling")
    parser = confidence_parsers.confidence_parser(parser)
    parser = confidence_parsers.sequential_parser(parser)
    parser = confidence_parsers.BM_parser(parser)
    parser = confidence_parsers.BPL_parser(parser)  # --help will show both BM and BPL
    parser = aircond.inparser_adder(parser)
    
    parser.add_argument("--solver-name",
                        help = "solver name (default gurobi)",
                        dest="solver_name",
                        type = str,
                        default="gurobi")

    parser.add_argument("--seed",
                        help="Seed for random numbers (default is 1134)",
                        dest="seed",
                        type=int,
                        default=1134)

    parser.add_argument("--BM-vs-BPL",
                        help="BM or BPL for Bayraksan and Morton or B and Pierre Louis",
                        dest="BM_vs_BPL",
                        type=str,
                        default=None)
    parser.add_argument("--xhat1-file",
                        help="File to which xhat1 should be (e.g. to process with zhat4hat.py)",
                        dest="xhat1_file",
                        type=str,
                        default=None)

    args = parser.parse_args()

    if args.BM_vs_BPL is None:
        raise argparse.ArgumentTypeError("--BM-vs_BPL must be given.")
    if args.BM_vs_BPL != "BM" and args.BM_vs_BPL != "BPL":
        raise argparse.ArgumentTypeError(f"--BM-vs_BPL must be BM or BPL (you gave {args.BM_vs_BMPL})")
    
    return args



if __name__ == '__main__':

    args = _parse_args()

    results = main(args)
    print(f"Final gap confidence interval results:", results)

    if args.xhat1_file is not None:
        print(f"Writing xhat1 to {args.xhat1_file}.npy")
        root_nonants =np.fromiter((v for v in results["Candidate_solution"]["ROOT"]), float)
        np.save(args.xhat1_file, root_nonants)
