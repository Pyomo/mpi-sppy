# Copyright 2020, 2021 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Use the aircond model to illustrate how to use sequential sampling.
#

import sys
import numpy as np
import argparse
import mpisppy.tests.examples.aircond as aircond
import pyomo.environ as pyo
import pyomo.common.config as pyofig
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgamator as amalgamator
import mpisppy.confidence_intervals.multi_seqsampling as multi_seqsampling
import mpisppy.confidence_intervals.confidence_config as conf_config
from mpisppy.utils import config

#============================
def xhat_generator_aircond(scenario_names, solvername="gurobi", solver_options=None,
                           branching_factors=None, mu_dev = 0, sigma_dev = 40,
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
    mu_dev: float, optional
        The average deviation of demand between two stages; The default is 0.
    sigma_dev: float, optional
        The standard deviation from mu_dev for the demand difference between
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
    
    cfg = config.Config()
    cfg.quick_assign("EF_mstage", bool, True)
    cfg.quick_assign("EF_solver_name", str, solvername)
    #cfg.quick_assign("solvername", str, solvername)  # amalgamator wants this
    cfg.quick_assign("EF_solver_options", dict, solver_options)
    cfg.quick_assign("num_scens", int, num_scens)
    cfg.quick_assign("_mpisppy_probability", float, 1/num_scens)
    cfg.quick_assign("start_seed", int, start_seed)
    cfg.add_and_assign("branching_factors", description="branching factors", domain=pyofig.ListOf(int), default=None, value=branching_factors)
    cfg.quick_assign("mu_dev", float, mu_dev)
    cfg.quick_assign("sigma_dev", float, sigma_dev)
    cfg.quick_assign("start_ups", bool, start_ups)

    #We use from_module to build easily an Amalgamator object
    ama = amalgamator.from_module("mpisppy.tests.examples.aircond",
                                  cfg, use_command_line=False)
    #Correcting the building by putting the right scenarios.
    ama.scenario_names = scenario_names
    ama.verbose = False
    ama.run()
    
    # get the xhat
    xhat = sputils.nonant_cache_from_ef(ama.ef)

    return {'ROOT': xhat['ROOT']}


def main(cfg):
    """ Code for aircond sequential sampling (in a function for easier testing)
    Uses the global config data.
    Returns:
        results (dict): the solution, gap confidence interval and T 
    """
    refmodelname = "mpisppy.tests.examples.aircond"
    scenario_creator = aircond.scenario_creator

    BFs = cfg.branching_factors
    num_scens = np.prod(BFs)

    scenario_creator_kwargs = {"num_scens" : num_scens,
                               "branching_factors": BFs,
                               "mu_dev": cfg.mu_dev,
                               "sigma_dev": cfg.sigma_dev,
                               "start_ups": cfg.start_ups,
                               "start_seed": cfg.seed,
                               }
    
    scenario_names = ['Scenario' + str(i) for i in range(num_scens)]
    
    xhat_gen_kwargs = {"scenario_names": scenario_names,
                        "solvername": cfg.solver_name,
                        "solver_options": None,
                        "branching_factors" : BFs,
                        "mu_dev": cfg.mu_dev,
                        "sigma_dev": cfg.sigma_dev,
                        "start_ups": False,
                        "start_seed": cfg.seed,
                        }

    cfg.quick_assign("xhat_gen_kwargs", dict, xhat_gen_kwargs)

    if cfg.BM_vs_BPL == "BM":
        # Bayraksan and Morton

        sampler = multi_seqsampling.IndepScens_SeqSampling(refmodelname,
                                          xhat_generator_aircond,
                                          cfg,
                                          stochastic_sampling=False,
                                          stopping_criterion="BM",
                                          solving_type="EF_mstage",
                                          )
    else:  # must be BPL
        ss = int(cfg.BPL_n0min) != 0
        sampler = multi_seqsampling.IndepScens_SeqSampling(refmodelname,
                                xhat_generator_aircond,
                                cfg,
                                stochastic_sampling=ss,
                                stopping_criterion="BPL",
                                solving_type="EF_mstage",
                                )
        
    xhat = sampler.run()
    return xhat

def _parse_args():
    # create a Config object and parse into it
    cfg = config.Config()
    cfg.multistage()
    conf_config.confidence_config(cfg)
    conf_config.sequential_config(cfg)
    conf_config.BM_config(cfg)
    conf_config.BPL_config(cfg)  # --help will show both BM and BPL

    aircond.inparser_adder(cfg)
    
    cfg.add_to_config("solver_name",
                      description = "solver name (e.g. gurobi)",
                      domain = str,
                      default=None)

    cfg.add_to_config("seed",
                      description="Seed for random numbers (default is 1134)",
                      domain=int,
                      default=1134)

    cfg.add_to_config("BM_vs_BPL",
                      description="BM or BPL for Bayraksan and Morton or B and Pierre Louis",
                      domain=str,
                      default=None)
    cfg.add_to_config("xhat1_file",
                      description="File to which xhat1 should be (e.g. to process with zhat4hat.py)",
                      domain=str,
                      default=None)

    cfg.parse_command_line("farmer_sequential")
    
    if cfg.BM_vs_BPL is None:
        raise RuntimeError("--BM-vs-BPL must be given.")
    if cfg.BM_vs_BPL != "BM" and cfg.BM_vs_BPL != "BPL":
        raise RuntimeError(f"--BM-vs-BPL must be BM or BPL (you gave {args.BM_vs_BMPL})")

    return cfg

if __name__ == '__main__':

    cfg = _parse_args()

    results = main(cfg)
    print(f"Final gap confidence interval results:", results)

    if cfg.xhat1_file is not None:
        print(f"Writing xhat1 to {cfg.xhat1_file}.npy")
        root_nonants =np.fromiter((v for v in results["Candidate_solution"]["ROOT"]), float)
        np.save(cfg.xhat1_file, root_nonants)
