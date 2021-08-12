# Copyright 2021 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
"""
An example of using amalgomator and solving directly the EF 
To execute this:
    python farmer_ama.py  --num-scens=10 --crops-multiplier=3 --farmer-with-integer
    
WARNING:
    num-scens must be specified !
"""
import mpisppy.utils.amalgomator as amalgomator

def main():
    solution_files = {"first_stage_solution":"farmer_first_stage.csv",
                      }
    ama_options = {"EF-2stage": True,   # We are solving directly the EF
                   "write_solution":solution_files}
    #The module can be a local file
    ama = amalgomator.from_module("afarmer", ama_options)
    ama.run()
    print("first_stage_solution=", ama.first_stage_solution)
    print("inner bound=", ama.best_inner_bound)
    print("outer bound=", ama.best_outer_bound)

if __name__ == "__main__":
    main()