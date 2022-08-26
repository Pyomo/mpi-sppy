# Copyright 2021 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
"""
An example of using amalgamator with 3 cylinders and one extension 
To execute this:
    mpiexec -np 3 python uc_ama.py --default-rho=1 --num-scens=3 --fixer-tol=1e-2
    
WARNING:
    num-scens must be taken as a valid value, i.e. among (3,5,10,25,50,100)
"""
import mpisppy.utils.amalgamator as amalgamator
from uc_funcs import id_fix_list_fct
from mpisppy.utils import config
import pyomo.common.config as pyofig

def main():
    solution_files = {"first_stage_solution":"uc_first_stage.csv",
                      #"tree_solution":"uc_ama_full_solution" 
                      #It takes too long to right the full solution
                      }
    cfg = config.Config()
    cfg.add_and_assign("id_fix_list_fct", "fct used by fixer extension", 
                                        domain=None, default=None,
                                        value = id_fix_list_fct)
    cfg.add_and_assign("2stage", description="2stage vsus mstage", domain=bool, default=None, value=True)
    cfg.add_and_assign("cylinders", description="list of cylinders", domain=pyofig.ListOf(str), default=None, value=['ph','xhatshuffle','lagranger'])
    cfg.add_and_assign("extensions", description="list of extensions", domain=pyofig.ListOf(str), default=None, value= ['fixer'])
    cfg.add_and_assign("write_solution", description="list of extensions", domain=None, default=None, value=solution_files)

    ama = amalgamator.from_module("uc_funcs", cfg)
    ama.run()
    if ama.on_hub:
        print("first_stage_solution=", ama.first_stage_solution)
        print("inner bound=", ama.best_inner_bound)
        print("outer bound=", ama.best_outer_bound)

if __name__ == "__main__":
    main()
