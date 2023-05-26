.. _zhat introduction:

zhat and Background for Confidence Intervals
============================================

In this section, we provide some reference material useful for
writing programs that compute confidence intervals.

xhat
----

Unless you are directly using mid-level functionality, your
code may need to write the root node nonanticipative variable values
(called `xhat` or `xhat_one`) to a file for later processing. This is
typically done using ``sputils.ef_ROOT_nonants_npy_serializer``, which
is shown in various examples, e.g., ``examples.farmer.farmer.py``

zhat4xhat
---------

The program ``zhat4xhat`` estimates approximate confidence intervals
for the objective function value, zhat, given an xhat. See
``examples.farmer.farmer_zhat.bash`` for a bash script that first
creates the xhat file, then computes an out-of-sample confidence
interval for it. Note: this program does not compute a confidence
interval for zstar, which is done using software documented in
:ref:`MMW Confidence Intervals`.
Note: at the time of this writing, `zhat4xhat` does
not support a starting scenario other than the first scenario, so
some care might be needed if you want to avoid including scenarios
used to compute xhat.


seedoffset
----------

Most of the confidence interval code assumes that is can pass in a
seed, particularly to the ``scenario_creator`` function so that
replicates can be obtained. See ``examples.farmer.farmer.py`` for an
example.

When only a small number of scenarios are available, the
``scenario_creator`` function may need to take care to avoid
attempting to access non-existent scenarios. If the data are provided
in files, then `seedoffset` is a bit of a misnomer, but it needs to be
added to scenarios numbers to get the scenario number. Note: to date,
there is no code in the confidence interval code-base that rescales
probabilities, so unlike the rest of `mpi-sppy`, present confidence
interval code assumes equally likely scenarios.
