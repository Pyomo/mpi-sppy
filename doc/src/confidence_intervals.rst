.. Confidence intervals:

MMW confidence interval
=======================

If we want to assess the quality of a given candidate solution ``xhat``, we could
try and evaluate the optimality gap, i.e. the gap between the value of the objective function
at ``xhat`` and the value of the solution to our problem.
The class ``MMWConfidenceIntervals`` compute an estimator of the optimality gap
as described in [mmw1999]_ (Section 3.2) and an asymptotic confidence interval for
this gap. 

We will document two steps in the process : finding a candidate solution ``xhat``, 
and evaluating it


Finding a candidate solution
----------------------------

Computing this confidence interval means that we need to find a solution to 
an approximate problem, and evaluate how good a solution to this approximate problem ``xhat`` is.
In order to use MMW, ``xhat`` must be written using one of two functions 
``ef_ROOT_nonants_npy_serializer`` or ``write_spin_the_wheel_first_stage_solution``.
These functions write ``xhat`` to a file and can be read using ``read_xhat``.

Computing a confidence interval
-------------------------------

The first step in computing a confidence interval is creating a ``MMWConfidenceIntervals`` object
that takes as an argument an ``xhat`` and options.
This object has a ``run`` method that returns a gap estimator and a confidence interval on the gap.

Example
-------

An example of use, with the ``farmer`` problem, can be found in the main of ``mmwci.py``.


Sequential sampling
===================

Similarly, given an confidence interval, one can try to find a candidate solution
 ``xhat`` such that its optimality gap has this confidence interval.
The class ``SeqSampling`` is implementing three procedures described in 
[bm2009]_ and [bpl2011]_. It takes as an input a method to generate
candidate solutions and options, and returns a ``xhat`` and a confidence interval on
its optimality gap. 

Examples of use with the ``farmer`` problem and seveeral options can be found in the main of ``seqsampling.py``.
