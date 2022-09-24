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
from mpisppy.utils import config
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgamator as amalgamator
import mpisppy.confidence_intervals.seqsampling as seqsampling
import mpisppy.confidence_intervals.confidence_config as confidence_config


#============================
def xhat_generator_farmer(scenario_names, solver_name=None, solver_options=None,
                          use_integer=False, crops_multiplier=1, start_seed=None):
    '''
    For sequential sampling.
    Takes scenario names as input and provide the best solution for the 
        approximate problem associated with the scenarios.
    Parameters
    ----------
    scenario_names: list of str
        Names of the scenario we use
    solver_name: str, optional
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

    cfg = config.Config()
    cfg.quick_assign("EF_2stage", bool, True)
    cfg.quick_assign("EF_solver_name", str, solver_name)
    #cfg.quick_assign("solver_name", str, solver_name)  # amalgamator wants this
    cfg.quick_assign("EF_solver_options", dict, solver_options)
    cfg.quick_assign("num_scens", int, num_scens)
    cfg.quick_assign("_mpisppy_probability", float, 1/num_scens)
    cfg.quick_assign("start_seed", int, start_seed)
    
    #We use from_module to build easily an Amalgamator object
    ama = amalgamator.from_module("farmer", cfg, use_command_line=False)
    #Correcting the building by putting the right scenarios.
    ama.scenario_names = scenario_names
    ama.verbose = False
    ama.run()
    
    # get the xhat
    xhat = sputils.nonant_cache_from_ef(ama.ef)

    return {'ROOT': xhat['ROOT']}
    


def main(cfg):
    """ Code for farmer sequential sampling (in a function for easier testing)
    Args:
        args (parseargs): the command line arguments object from parseargs
    Returns:
        results (dict): the solution, gap confidence interval and T 
    """
    refmodelname = "farmer"
    scenario_creator = farmer.scenario_creator

    scen_count = cfg.num_scens
    assert cfg.EF_solver_name is not None
    solver_name = cfg.EF_solver_name
    crops_multiplier = cfg.crops_multiplier
    
    scenario_names = ['Scenario' + str(i) for i in range(scen_count)]
    
    xhat_gen_kwargs = {"scenario_names": scenario_names,
                       "solver_name": solver_name,
                       "solver_options": None,
                       "use_integer": False,
                       "crops_multiplier": crops_multiplier,
                       "start_seed": 0,
    }
    cfg.quick_assign("xhat_gen_kwargs", dict, xhat_gen_kwargs)

    # Note that as of July 2022, we are not using conditional args so cfg has everything
    if cfg.BM_vs_BPL == "BM":
        sampler = seqsampling.SeqSampling(refmodelname,
                                xhat_generator_farmer,
                                cfg,
                                stochastic_sampling=False,
                                stopping_criterion="BM",
                                solving_type="EF_2stage",
                                )
    else:  # must be BPL
        ss = int(cfg.BPL_n0min) != 0
        sampler = seqsampling.SeqSampling(refmodelname,
                                xhat_generator_farmer,
                                cfg,
                                stochastic_sampling=ss,
                                stopping_criterion="BPL",
                                solving_type="EF_2stage",
                                )
    xhat = sampler.run()
    return xhat

def _parse_args():
    # create a config object and parse
    cfg = config.Config()
    
    cfg.num_scens_required()

    farmer.inparser_adder(cfg)
    cfg.EF2()
    confidence_config.confidence_config(cfg)
    confidence_config.sequential_config(cfg)
    confidence_config.BM_config(cfg)
    confidence_config.BPL_config(cfg)  # --help will show both BM and BPL

    cfg.add_to_config("BM_vs_BPL",
                      description="BM or BPL for Bayraksan and Morton or B and Pierre Louis",
                      domain=str,
                      default=None)
    cfg.add_to_config("xhat1_file",
                      description="File to which xhat1 should be (e.g. to process with zhat4hat.py)",
                      domain=str,
                      default=None)

    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()    
    cfg.aph_args()    
    cfg.xhatlooper_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.lagranger_args()
    cfg.xhatshuffle_args()
    cfg.add_to_config("use_norm_rho_updater",
                         description="Use the norm rho updater extension",
                         domain=bool,
                         default=False)
    cfg.add_to_config("use-norm-rho-converger",
                         description="Use the norm rho converger",
                         domain=bool,
                         default=False)
    cfg.add_to_config("run_async",
                         description="Run with async projective hedging instead of progressive hedging",
                         domain=bool,
                         default=False)
    cfg.add_to_config("use_norm_rho_converger",
                         description="Use the norm rho converger",
                         domain=bool,
                         default=False)

    cfg.parse_command_line("farmer_sequential")

    if cfg.BM_vs_BPL is None:
        raise RuntimeError("--BM-vs_BPL must be given.")
    if cfg.BM_vs_BPL != "BM" and cfg.BM_vs_BPL != "BPL":
        raise RuntimeError(f"--BM-vs_BPL must be BM or BPL (you gave {cfg.BM_vs_BMPL})")
    
    return cfg


if __name__ == '__main__':

    cfg = _parse_args()
    
    results = main(cfg)
    print(f"Final gap confidence interval results:", results)

    if cfg.xhat1_file is not None:
        print(f"Writing xhat1 to {cfg.xhat1_file}.npy")
        root_nonants =np.fromiter((v for v in results["Candidate_solution"]["ROOT"]), float)
        np.save(cfg.xhat1_file, root_nonants)
