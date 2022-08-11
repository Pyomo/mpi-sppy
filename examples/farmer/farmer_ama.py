# Copyright 2021 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
"""
An example of using amalgamator and solving directly the EF 
To execute this:
    python farmer_ama.py  --num-scens=10 --crops-multiplier=3 --farmer-with-integer
    
WARNING:
    num-scens must be specified !
"""
import mpisppy.utils.amalgamator as amalgamator
from mpisppy.utils import config

def main():

    cfg = config.Config()
    cfg.add_and_assign("EF_2stage", description="EF 2stage vsus mstage", domain=bool, default=None, value=True)
    cfg.add_and_assign("first_stage_solution_csv", description="where to write soln", domain=str, default=None, value="farmer_first_stage.csv")
    
    #The module can be a local file
    ama = amalgamator.from_module("farmer", cfg)
    ama.run()
    print("first_stage_solution=", ama.first_stage_solution)
    print("inner bound=", ama.best_inner_bound)
    print("outer bound=", ama.best_outer_bound)

if __name__ == "__main__":
    main()
