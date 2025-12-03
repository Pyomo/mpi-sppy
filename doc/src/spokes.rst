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


Subgradient
^^^^^^^^^^^

This bound is based on the subgradient method. It computes its own W values
based on the Lagrangian relaxation and reports a bound to the hub.


Lagranger
^^^^^^^^^

This bounder is no longer recommended for use. It does not seem to work as well
as others.

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

ph_ob
^^^^^

This bounder is similar to the Lagrangian bounder, except that it executes a PH
algorithm to obtain its own W values. The idea is that it can use lower rho values
so as to obtain better outer (lower when minimizing) bounds. It can also provide
a Lagrangian bound even if the hub does not provide lagrangian multipliers.

The easiest way to use ``ph_ob`` is via the vanilla ``ph_ob_spoke`` method
as illustrated in ``examples.farmer_cylinders.py``. This method takes values
from the config object (assuming the config object's ``ph_ob`` method
was called as shown in the function ``examples.farmer_cylinders._parse_args``)
and sets up the options for the spoke.

The option ``ph-ob-initial-rho-rescale-factor`` defaults to 0.1, so if nothing
other than ``--ph-ob`` is given on the command line, the ph_ob spoke will use
one tenth the default rho (it might use one tenth of rho from
a rho setter if one is configured in the cylinders program and passed to the ph_ob
constructor). Additional control over rho values
is provided by the ``phob-rho-rescale-factors-json`` option which is a json
file that provides a dictionary with keys that are iteration numbers and values
that are rescale factors. Note that all rescaling is cummulative.

See ``examples.uc.gradient_uc_cylinders.py`` for an example that uses a cost-based
rho setter for the uc problem in the ph_ob cylinder.

As of August, 2024 use of a gradient based rho with ph_ob is untested.

Reduced Costs
^^^^^^^^^^^^^

The reduced cost spoke is equivalent to the Lagrangian spoke, except that it relaxes all 
integrality contraints in the subproblems. This enables the computation of reduced costs
for the first stage variables, which can be used for bound tightening or heuristic fixing
in the hub.


Inner Bounds
------------

For minimization problems, `inner bound` means `upper bound`. But more
importantly, the bounds are based on a solution, whose value can be
computed. In some sense, if you don't have this solution, you don't
have anything (even if you think your hub algorithm has `converged` in
some sense). We refer to this solution as xhat (:math:`\hat{x}`) or
somes as an `incumbent` solution.
`
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

xhatxbar_bounder
^^^^^^^^^^^^^^^^

This bounder computes and uses :math:`\overline{x}` as :math:`\hat{x}`. It does simple rounding
for integer variables.

stage2ef
~~~~~~~~

An option for ``xhatshufflelooper_bounder`` is under development 
for multistage problems that creates an EF for each second stage nodes by
fixing the first stage nonanticipative variables.  This code requires
that the number of ranks allocated to the ``xhatshufflelooper_bounder``
is an integer multiple of the number of second stage nodes. Here is a 
hint about how to to use it in a driver:

::

    xhatshuffle_spoke["opt_kwargs"]["options"]["stage2EFsolvern"] = solver_name
    xhatshuffle_spoke["opt_kwargs"]["options"]["branching_factors"] = branching_factors

An example is shown in ``examples.hydro.hydro_cylinders.py`` (this particular example
is intended to show the coding, not normal behavior. It is sort of an edge case:
including this option causes the upper bound to immediately be Z*)

 
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
