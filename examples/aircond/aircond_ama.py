# Copyright 2021 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
"""
An example of using amalgomator with 3 cylinders for a multistage problem
To execute this:
    mpiexec -np 3 python aircond_ama.py --default-rho=1 --branching-factors 3 3
    
WARNING:
    do not use the num-scens argument on the command line
"""
import numpy as np
import mpisppy.utils.amalgomator as amalgomator
import aaircond

def main():
    solution_files = {"first_stage_solution":"aircond_first_stage.csv",
                      "tree_solution":"aircond_full_solution" 
                      }
    options = {"mstage": True,   # 2stage vs. mstage
                   "cylinders": ['ph','xhatshuffle','lagrangian'],
                   "write_solution": solution_files
                   }
    ama_options = amalgomator.Amalgomator_parser(options, aaircond.inparser_adder)
    
    #Here, we need to change something specified by the command line
    #That's why we cannot use amalgomator.from_module
    if ama_options['num_scens'] is not None:
        raise RuntimeError("Do not use num_scens here, we want to solve the problem for the whole sample scenario tree")
    
    num_scens = np.prod(ama_options['branching_factors'])
    scenario_names = aaircond.scenario_names_creator(num_scens,0)
    
    ama =amalgomator.Amalgomator(ama_options, 
                                 scenario_names, 
                                 aaircond.scenario_creator, 
                                 aaircond.kw_creator)
    ama.run()
    if ama.on_hub:
        print("first_stage_solution=", ama.first_stage_solution)
        print("inner bound=", ama.best_inner_bound)
        print("outer bound=", ama.best_outer_bound)

if __name__ == "__main__":
    main()