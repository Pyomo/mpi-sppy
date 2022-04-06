.. _sec-aph:

APH
===

The code is based on "Algorithm 2: Asynchronous projective hedging
(APH) -- Algorithm 1 specialize to the setup S1-S4" from "Asynchronous
Projective Hedging for Stochastic Programming"
http://www.optimization-online.org/DB_HTML/2018/10/6895.html

Options
^^^^^^^

The following table lists the simple options. The column labeled ``Paper``
gives an indication of what parameter in the paper is related to the
parameter in the software (the parameters with Greek letters are one-to-one
and the other are not). The column
labeled ``PHoptions`` gives the name in the options dictionary
passed to the APH constructor. The column labeled ``Parser`` gives the
name used in args parser for examples.

.. list-table:: APH Options
   :widths: 10 15 15 30
   :header-rows: 1

   * - Paper
     - PHoptions
     - Parser
     - Description
   * - ...
     - "PHIterLimit"
     - --max-iterations
     - A termination criterion
   * - :math:`\rho`
     - "defaultPHrho"
     - --default-rho
     - Proximal term multiplier in objective and y update
   * - :math:`\nu`
     - "APHnu"
     - --aph-nu
     - Step size multiplier (e.g. 0.5; default 1)
   * - :math:`\gamma`
     - "APHgamma"
     - --aph-gamma
     - Primal vs. dual emphasis (0,2) default 1; larger is primal
   * - :math:`I_{k}`
     - "dispatch_frac"
     - --dispatch-frac
     - Fraction of subproblems at each rank to dispatch
   * - :math:`I_{k}`
     - "async_frac_needed"
     - --aph-frac-needed
     - Fraction of ranks to wait for (default 1)
   * -
     - "async_sleep_secs"
     - --listener-sleep-secs
     - The software hangs if too small (default 0.5)
   * - :math:`d(i,k)`
     - "APHuse_lag"
     - --with-aph-lag
     - In step 7,8 use w and z from last solve for i
       
Most of these could be varied iteration by iteration using callback extensions.

x-hat
^^^^^

In addition to the notation in the paper, the software has the concept of
x-hat, which is a solution that is implementable and admissible (feasible
for all scenarios). Asynchronously and in parallel with APH there is
software that continuously uses the x values found by APH to evaluate
as potential best x-hat values.

The current xhat software is not specialized for APH. For APH with
very low dispatch fractions, one might want to also look at z as
a candidate for x-hat for problems where feasibility is not an issue.

The software `xhatshuffelooper` looks at maybe four or five x values per
PH iteration, and fewer per APH iteration with dispatch fractions below 1.

Convergence Metric
^^^^^^^^^^^^^^^^^^

At each iteration, the software outputs a convergence metric that is

.. math::

   \frac{||u||_{2}^{2}}{||w||_{2}^{2}} + \frac{||v||_{2}^{2}}{||z||_{2}^{2}}
   
where the norms are probability weighted.

Dispatch
^^^^^^^^

Dispatch is based on most negative :math:`\phi` and if there are not
enough negative :math:`\phi`, then the least recently dispatched are
dispatched to fill out the dispatch fraction.

farmer
^^^^^^

The scripts for this example are currently in the paper repo in
`AsyncPH/experiments/challange/farmer`; the driver is
`farmer_driver.py`.  The driver references the model, which is in the
`mpi-sppy` repo.  The `aph05.bash` script is intended
to have a dispatch fraction of 1/2 (hence the 05 for 0.5 in the name).
Aside: you can run PH with `quicky.bash`.

ranks
-----

The mpi-sspy software wants to know how many bundles per rank (0 means
no bundles).  Meanwhile, mpiexec needs to know how many total
ranks. For the farmer example, the only spoke is for xhat, so you need
twice as many ranks for mpiexec in total as will be allocated to APH.

A Peek Under the Hood
^^^^^^^^^^^^^^^^^^^^^

The APH implementation has a listener thread that continuously does
MPI reductions and a worker thread that does most of the work. A wrinkle
is that the listenter thread does a `side gig` if enough ranks have reported
(the "async_frac_needed" option) because after it has
done the reductions to get u and v and needs to do some calculations, and
then reductions to compute tau and theta.
It turns out to have been easier to implement the general notion of side
gigs in the general-purpose listener software than it would have been
to something special purpose.  The side gig that is implemented for
APH checks to see if there are enough ranks reporting.

Lags are supported with if blocks that control how the objective
function is constructed and how y is updated.
