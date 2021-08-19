.. _Confidence intervals:

MMW confidence interval
=======================

If we want to assess the quality of a given candidate solution ``xhat_one`` 
(a first stage policy), we could try and evaluate the optimality gap, i.e. 
the gap between the value of the objective function
at ``xhat_one`` and the value of the solution to our problem.
The class ``MMWConfidenceIntervals`` compute an estimator of the optimality gap
as described in [mmw1999]_ (Section 3.2) and an asymptotic confidence interval for
this gap. 

We will document two steps in the process : finding a candidate solution 
``xhat_one``, and evaluating it.


Finding a candidate solution
----------------------------

Computing this confidence interval means that we need to find a solution to 
an approximate problem, and evaluate how good a solution to this approximate problem ``xhat_one`` is.
In order to use MMW, ``xhat_one`` must be written using one of two functions 
``ef_ROOT_nonants_npy_serializer`` or ``write_spin_the_wheel_first_stage_solution``.
These functions write ``xhat`` to a file and can be read using ``read_xhat``.
When using a cylinders driver, the function ``sputils.first_stage_nonant_npy_serializer``
can be given as the ``first_stage_solution_writer`` argument to the function
``sputils.write_spin_the_wheel_first_stage_solution``. See the ``farmer_cylinders.py`` example.

Evaluating a candidate solution
-------------------------------

To evaluate a candidate solution with some scenarios, one might
create a ``Xhat_Eval`` object and call its ``evaluate`` method 
(resp. ``evaluate_one`` for a single scenario). It takes as
an argument ``xhats``, a dictionnary of noon-anticipative policies for all 
non-leaf nodes of a scenario tree. While for a 2-stage problem, ``xhats`` is
just the candidate solution ``xhat_one``, for multistage problem the 
dictionnary can be computed using the function ``walking_tree_xhats`` 
(resp. ``feasible_solution``).


Computing a confidence interval
-------------------------------

The first step in computing a confidence interval is creating a ``MMWConfidenceIntervals`` object
that takes as an argument an ``xhat_one`` and options.
This object has a ``run`` method that returns a gap estimator and a confidence interval on the gap.

Example
-------

An example of use, with the ``farmer`` problem, can be found in the main of ``mmwci.py``.

Using stand alone ``mmw_conf.py``
---------------------------------

(Currently for use with 2-stage problem only)

``mmw_conf`` uses the ``MMWConfidenceIntervals`` class from ``mmw_ci`` in order to construct a confidence interval on the optimality gap of a particular candidate solution ``xhat`` of a model instance. 

To use the stand along program a model compatible with ``Amalgomator`` and ``.npy`` file with a candidate solution to an instance of the model are required.

First, assume that the model to be used is compatable with the ``Amalgomator`` class. This requires the model to have each of the following: a ``scenario_names_creator``,  a ``scenario_creator``, an ``inparser_adder``, and a ``kw_creator``. See ``afarmer.py`` in ``examples`` for an example of an acceptable model.

Once a model satisfies the requirement for amalgomator, next a ``.npy`` file should be constructed from the given model. This can be accomplished, for example, by adding the line 
``sputils.ef_ROOT_nonants_npy_serializer(instance, 'xhat.npy')`` after solving the ef ``instance``. When using ``Amalgomator`` to solve the program, this can be done by adding the line
``sputils.ef_ROOT_nonants_npy_serializer(ama_object.ef, "xhat.npy")`` to your existing program (see the example in ``afarmer.py`` for an example of this).

Once this is accomplished, on the command line, run
``python -m mpisppy.confidence_intervals.mmw_conf my_model.py xhat.npy gurobi --num-scens n --alpha 0.95``. Note that ``xhat.npy`` is assumed to be in the same directory as ``my_model.py`` in this case. If the file is saved elsewhere then the corresponing path should be called on the command line.

Additional solver options can be specified with the ``--solver-options`` option.

This program will out put a confidence interval on the gap between the solution to the EF and the optimal solution. There is an additional option, ``--with-objective-gap``, which will computes a confidence interval around the solution of the stochastic program. Since the exact value of the objective function cannot be determined, we use the realizations of the objective function at the candidate solution to construct an additional confidence interval about the mean of the realizations computed.

Sequential sampling
===================

Similarly, given an confidence interval, one can try to find a candidate solution
``xhat_one`` such that its optimality gap has this confidence interval.
The class ``SeqSampling`` implements three procedures described in 
[bm2011]_ and [bpl2012]_. It takes as an input a method to generate
candidate solutions and options, and returns a ``xhat_one`` and a confidence interval on
its optimality gap.

Examples of use with the ``farmer`` problem and several options can be found in the main of ``seqsampling.py``.

.. Note::
   Unlike MMW, sequential samping does not run with multistage problems.

