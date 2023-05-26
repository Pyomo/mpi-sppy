CCOPF
=====
Contingency Constrained OPF

This example is not fully developed as of October 2020. As of May 2022, it executes
but uses mid- and lower-level code.

To use this, egret must be installed (also run
egret/thirdparty/get_pglib_opf.py and then you still
might have to make sure the directories match.)


EF
---

- To get the EF use ``python ccopf_multistage.py bf1 bf2 bf3
  where bf1 and bf2 are branching factors (e.g. 2 3 1 for a small test).
Use 1 and 0 for iters and bundles
for a small test use
python ccopf_multistage.py 2 3 1

  Edit the line in the py file that assigns `casename` to change the example that is run.

PH
--
  
- To run with PH and a fixed scenario set for the upper bound use
  ``mpiexec -np 2 python -m mpi4py ccopf2wood.py bf1 bf2 iters scenperbund solver``
  e.g.,
  ``mpiexec -np 2 python -m mpi4py ccopf2wood.py 2 3 2 0 cplex``

The number of processors is restricted by the branching factors; basically, multiples of the
first branching factor.

  Edit the line in the py file that assigns `casename` to change the example that is run.


  
To change other parameters, change these lines in ccopf2wood.py:
    # start options
    solver_name = "ipopt"
    number_of_stages = 3
    stage_duration_minutes = [5, 15, 30]
    seed = 1134
    a_line_fails_prob = 0.2
    repair_fct = FixFast
    # end options

Notes about some of these options:

    - egret_path_to_data: starts from Egret/egret
    - stage_duration_minutes: would be used by a non-trivial repair function
    - branching_factors: there is no branching factor for the leaf nodes
    - a_line_fails_prob: this is the probability that a random line fails

One more note:
     the repair function presently hard-wired is called FixFast, which
     just brings the line back up in the next stage

.. note::

   As of Dec 2019, this example has only an inner bound (an outer bound
   would take work because this is a non-convex problem solved by ipopt.
