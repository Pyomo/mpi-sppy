###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Use the aircond model to illustrate how to use sequential sampling.
#

import numpy as np
import mpisppy.tests.examples.aircond as aircond
import mpisppy.confidence_intervals.multi_seqsampling as multi_seqsampling
import mpisppy.confidence_intervals.confidence_config as conf_config
from mpisppy.utils import config


def main(cfg):
    """ Code for aircond sequential sampling (in a function for easier testing)
    Uses the global config data.
    Returns:
        results (dict): the solution, gap confidence interval and T 
    """
    refmodelname = "mpisppy.tests.examples.aircond"

    BFs = cfg.branching_factors
    num_scens = np.prod(BFs)

    scenario_names = ['Scenario' + str(i) for i in range(num_scens)]
    
    xhat_gen_kwargs = {"scenario_names": scenario_names,
                        "solver_name": cfg.solver_name,
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
                                          aircond.xhat_generator_aircond,
                                          cfg,
                                          stochastic_sampling=False,
                                          stopping_criterion="BM",
                                          solving_type="EF_mstage",
                                          )
    else:  # must be BPL
        ss = int(cfg.BPL_n0min) != 0
        sampler = multi_seqsampling.IndepScens_SeqSampling(refmodelname,
                                aircond.xhat_generator_aircond,
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
        raise RuntimeError(f"--BM-vs-BPL must be BM or BPL (you gave {cfg.BM_vs_BPL})")

    return cfg

if __name__ == '__main__':

    cfg = _parse_args()
    cfg.quick_assign("EF_mstage", bool, True)

    results = main(cfg)
    print("Final gap confidence interval results:", results)

    if cfg.xhat1_file is not None:
        print(f"Writing xhat1 to {cfg.xhat1_file}.npy")
        root_nonants =np.fromiter((v for v in results["Candidate_solution"]["ROOT"]), float)
        np.save(cfg.xhat1_file, root_nonants)
