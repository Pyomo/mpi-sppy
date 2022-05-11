# Copyright 2021 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
"""
An example of using amalgamator with 3 cylinders for a multistage problem
To execute this:
    mpiexec -np 3 python aircond_ama.py --default-rho=1 --branching-factors "3 3" --lagrangian --xhatshuffle
    
WARNING:
    do not use the num-scens argument on the command line
"""
import numpy as np
import mpisppy.utils.amalgamator as amalgamator

import mpisppy.tests.examples.aircond as aircond

def main():
    solution_files = {"first_stage_solution":"aircond_first_stage.csv",
                      "tree_solution":"aircond_full_solution" 
                      }
    options = {"mstage": True,   # 2stage vs. mstage
                   "cylinders": ['ph','xhatshuffle','lagrangian'],
                   "write_solution": solution_files
                   }
    ama_options = amalgamator.Amalgamator_parser(options, aircond.inparser_adder)
    
    #Here, we need to change something specified by the command line
    #That's why we cannot use amalgamator.from_module
    if ama_options.get('num_scens', None) is not None:
        raise RuntimeError("Do not use num_scens here, we want to solve the problem for the whole sample scenario tree")
    
    num_scens = np.prod(ama_options['branching_factors'])
    scenario_names = aircond.scenario_names_creator(num_scens,0)
    
    ama =amalgamator.Amalgamator(ama_options, 
                                 scenario_names, 
                                 aircond.scenario_creator, 
                                 aircond.kw_creator)
    ama.run()
    if ama.on_hub:
        print("first_stage_solution=", ama.first_stage_solution)
        print("inner bound=", ama.best_inner_bound)
        print("outer bound=", ama.best_outer_bound)

if __name__ == "__main__":
    main()
