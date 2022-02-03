.. _Spokes:

Spokes
======

In this section we provide an overview of some of the spoke classes
that are part of the ``mpi-sppy`` release.


Outer Bound
-----------

For minimization problems, `outer bound` means `lower bound`.

Frank-Wolfe
^^^^^^^^^^^

This bound is based on the paper `Combining Progressive Hedging with a
Frank--Wolfe Method to Compute Lagrangian Dual Bounds in Stochastic
Mixed-Integer Programming` by Boland et al [boland2018]_. It does not receive
information from the hub, it simply sends bounds as they are available.
Compared to the Lagrangian bounds, it takes longer to compute but is generally
tighter once it reports a bound.


Lagrangian
^^^^^^^^^^

This bound is based on the paper `Obtaining lower bounds from the progressive
hedging algorithm for stochastic mixed-integer programs` by Gade et al
[gade2016]_. It takes W values from the hub and uses them to compute a bound.


Lagranger
^^^^^^^^^

This bound is a variant of the Lagrangian bound, but it takes x values from the
hub and uses those to compute its own W values. It can modify the rho
values (typically to use lower values). The modification is done
in the form of scaling factors that are specified to be applied at a given
iteration. The factors accumulate so if 0.5 is applied at iteration 1 and
1.2 is applied at iteration 10, from iteration 10 onward, the factor will be 0.6. Here
is a sample json file:

::
   
   {
    "1": "0.5",
    "10": "1.2"
   }




Inner Bounds
------------

For minimization problems, `inner bound` means `upper bound`. But more
importantly, the bounds are based on a solution, whose value can be
computed. In some sense, if you don't have this solution, you don't
have anything (even if you think your hub algorithm has `converged` in
some sense). We refer to this solution as xhat (:math:`\hat{x}`)

xhat_specific_bounder
^^^^^^^^^^^^^^^^^^^^^

At construction, this spoke takes a specification of a scenario per
non-leaf node of the scenario tree (so for a two-stage problem, one
scenario), which are used at every iteration of the hub algorithm as
trial values for :math:`\hat{x}`.

xhatshufflelooper_bounder
^^^^^^^^^^^^^^^^^^^^^^^^^

This bounder shuffles the scenarios and loops over them to try a 
:math:`\hat{x}` until
the hub provides a new x.  To ensure that all subproblems are tried
eventually, the spoke remembers where it left off, and resumes from
its prior position.  Since the resulting subproblems after fixing the
first-stage variables are usually much easier to solve, many candidate
solutions can be tried before receiving new x values from the hub.

This spoke also supports multistage problems. It does not try every subproblem, but
shuffles the scenarios and loops over the shuffled list.
At each step, it takes the first-stage solution specified by a scenario, 
and then uses the scenarios that follows in the shuffled loop to get the 
values of the non-first-stage variables that were not fixed before.
 
slam_heuristic
^^^^^^^^^^^^^^

This heuristic attempts to find a feasible solution by slamming every
variable to its maximum (or minimum) over every scenario associated 
with that scenario tree node. This spokes only supports two-stage problems at this time.


General
-------

cross scenario
^^^^^^^^^^^^^^

Passes cross scenario cuts.


spoke_sleep_time
----------------

This is an advanced topic and rarely encountered.
In some settings, particularly with small sub-problems, it is possible for
ranks within spokes to become of of sync.  The most common manifestation of this
is that some ranks do not see the kill signal and sit in a busy-wait I/O loop
until something external kills them; but it can also be the case that Lagrangian
bound spokes start operating on data from different hub iterations; they should notice
this an emit a message if it happens.

This problem is normally avoided by default actions in lower level code (in `spcommunicator.py`)
that insert a short sleep. To compute the sleep duration, it uses a heuristic based on the
number of non-anticipative variables. It is also possible to explicitly set this sleep time.
At the lowest levels, this is done by setting a value for "spoke_sleep_time" in the options
dictionary passed to the ``SPCommunicator`` constructor. At a higher level, it is possible
to pass a `spoke_sleep_time` keyword argument to the vanilla hub and spoke constructors. This
is illustrated in `hydro_cylinders.py` example (in the `hyrdo` example directory). You
should probably pass the same value to all constructors.
