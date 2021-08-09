# Copyright 2021 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
"""
An example of using amalgomator with 3 cylinders and one extension 
To execute this:
    mpiexec -np 3 python uc_ama.py --default-rho=1 --num-scens=3 --fixer-tol=1e-2
"""
import mpisppy.utils.amalgomator as amalgomator
from uc_funcs import id_fix_list_fct

def main():
    solution_files = {"first_stage_solution":"uc_first_stage.csv",
                      #"tree_solution":"uc_ama_full_solution" It takes too long to right the full solution
                      }
    ama_options = {"2stage": True,   # 2stage vs. mstage
                   "cylinders": ['ph','xhatshuffle','lagrangian'],
                   "extensions": ['fixer'],
                   "id_fix_list_fct": id_fix_list_fct, #Needed for fixer, but not passed in baseparsers
                   "write_solution":solution_files
                   }
    ama = amalgomator.from_module("uc_funcs", ama_options)
    ama.run()
    # print(f"inner bound=", ama.best_inner_bound)
    # print(f"outer bound=", ama.best_outer_bound)

if __name__ == "__main__":
    main()