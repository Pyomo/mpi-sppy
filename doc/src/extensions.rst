.. _Extensions:

Extensions
==========

In order to allow for extension or modification of code behavior, many of
the components (hubs and spokes) support callout points. The objects
that provide the methods that are called are referred to as `extensions`.
Instead of using an extension, you could just hack in a function call,
but then every time ``mpi-sppy`` is updated, you would have to remember
to hack it back in. By using an extension object, the addition
or modification will remain available. Perhaps more important, the
extension can be included in some applications, but not others.

There are a number of extensions, particularly for PH, that are provided
with ``mpi-sppy`` and they provide examples that can be used for the
creation of more. Extensions can be found in ``mpisppy.extensions``.
Note that some things (e.g. some xhatters) can be used as a cylinder
or as an extension. A few other things (e.g., cross scenario cuts) need
both an extension and a cylinder.

Many extensions are supported in :ref:`generic_cylinders`. The rest of
this help file describes extensions released with mpisppy along with
some hints for including them in your own cylinders driver program.

Multiple Extensions
-------------------

To employ multiple PH extensions, use ``mpisppy.extensions.extension import MultiExtension``
that allows you to give a list of extensions that will fire in order
at each callout point. See, e.g. ``examples.sizes.sizes_demo.py`` or
``examples.farmer.farmer_rho_demo.py`` for an
example of use.

.. note::
   The ``MultiExtension`` constructor in ``mpisppy.extensions.extensions.py``
   takes a list of extensions classes in addition to the optimization object
   (e.g. inherited from ``PHBase``). However, ``cfg_vanilla.py`` wants
   to see the class ``MultiExtension`` in the hub or cylinder dict entry
   for ``["opt_kwargs"]["extensions"]`` and then wants to see a list of
   extension classes in ``["opt_kwargs"]["extension_kwargs"]["ext_classes"]``.
   Some examples do both, which can be little confusing.


PH extensions
-------------

Some of these can be used with other hubs. An extension object can be
passed to the PH constructor and it is assumed to have methods defined
for all the callout points in PH (so all of the examples do). To see 
the callout points look at ``phbase.py``. Extensions can also specify
callout points in the `Hub` `SPCommunicator` object: these callout points
are especially useful for writing custom `Spoke` objects which can then
interact with the hub PH object. To see the callout points look at
``cylinders/hub.py``; an example of such an extension is the
"cross-scenario cut" extension defined in ``extensions/cross_scen_extension.py``
and associated spoke object defined in ``cylinders/cross_scen_spoke.py``.

If you want to use more than one extension, define a main extension that has
a reference to the other extensions and can call their methods in the
appropriate order. Extensions typically access low level elements of
``mpi-sppy`` so writing your extensions is an advanced topic. We will
now describe a few of the extensions in the release.

mipgapper.py
^^^^^^^^^^^^

This is a good extension to look at as a first example. It takes a
dictionary with iteration numbers and mipgaps as input and changes the
mipgap at the corresponding iterations. The dictionary is provided in
the options dictionary in ``["gapperoptions"]["mipgapdict"]``.  There
is an example of its use in ``examples.sizes.sizes_demo.py``.

Instead of an options dictionary, when run with cylinders the options
``["gapperoptions"]["starting_mipgap"]`` and ``["gapperoptions"]["mipgap_ratio"]``
can be set. The ``starting_mipgap`` will be the initial value used,
and as the cylinders close the relative optimality gap the extension will set the subproblem
mipgaps as the ``min(starting_mipgap, mipgap_ratio * problem_ratio)``, where
the ``problem_ratio`` is the relative optimality gap on the overall problem
as computed by the cylinders.

This extension can also be used with the Lagrangian and subgradient spokes.

fixer.py
^^^^^^^^

This extension provides methods for fixing nonanticipative variables (usually integers) for
which all scenarios have agreed for some number of iterations. There
is an example of its use in ``examples.sizes.sizes_demo.py`` also
in ``examples.sizes.uc_ama.py``. The ``uc_ama`` example illustrates
that when ``amgalgamator`` is used ``"id_fix_list_fct"`` needs
to be on the ``Config`` object so the amalgamator can find it.

.. note::

   For the iteration zero fixer tuples, the iteration counts are just
   compared with None. If you provide a count for iteration zero, the
   variable will be fixed if it is within the tolerance of being converged.
   So if you don't want to fix a variable at iteration zero, provide a
   tolerance, but set all count values to ``None``.

reduced_cost_fixer
^^^^^^^^^^^^^^^^^^

This extension provides methods for fixing nonanticipative variables based on their expected
reduced cost as calculated by the ReducedCostSpoke. The aggressiveness of the
fixing can be controled through the ``zero_rc_tol`` parameter (reduced costs
with magnitude below this value will be considered 0 and not eligible for fixing)
and the ``fix_fraction_target`` paramemters, which set a maximum fraction of
nonanticipative variables to be fixed based on expected reduced costs. These two
parameters iteract with each other -- the expected reduced costs are sorted by
magnitude, and if the `fix_fraction_target`` percental is below ``zero_rc_tol``,
then fewer than ``fix_fraction_target`` variables will be fixed. Further, to
have a defined expected reduced cost, all nonant variable values *must be* at
the same bound in the ReducedCostSpoke.

Variables will be unfixed if they no longer meet the expected reduced cost
criterion for fixing, e.g., the variable's expected reduced cost became too
low or the variable was not at its bound in every subproblem in the ReducedCostSpoke.

relaxed_ph_fixer
^^^^^^^^^^^^^^^^

This extension will fix nonanticipative variables at their bound if they are at
their bound in the RelaxedPHSpoke for that subproblem. It will similarily unfix
nonanticipative variables which are not at their bounds in the RelaxedPHSpoke.
Because different nonanticipative variables are fixed in different suproblems,
it will also unfix nonanticipative variables if their value is *not* at the the current
consensus solution xbar (because the variable was not fixed in a different subproblem
and therefore came off its bound).

xhat
^^^^

Most of the xhat methods can be used as an extension instead of being used
as a spoke, when that is desired (e.g. for serial applications).

integer_relax_then_enforce
^^^^^^^^^^^^^^^^^^^^^^^^^^

This extension is for problems with integer variables. The scenario subproblems
have the integrality restrictions initially relaxed, and then at a later point
the subproblem integrality restrictions are re-enabled. The parameter ``ratio``
(default = 0.5) controls how much of the progressive hedging algorithm, either
in the iteration or time limit, is used for relaxed progressive hedging iterations.
The extension will also re-enforce the integrality restrictions if the convergence
threshold is within 10\%  of the convergence tolerance.

This extension can be especially effective if (1) solving the relaxation
is much easier than solving the problem with integrality constraints or (2) the
relaxation is reasonably "tight".

WXBarWriter and WXBarReader
^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is an extension to write xbar and W values and another to read them.
An example of their use is shown in ``examples.sizes.sizes_demo.py``

norm_rho_updater
^^^^^^^^^^^^^^^^

This extension adjust rho dynamically. The code is in ``mpisppy.extensions.norm_rho_updater.py``
and there is an accompanying converger in ``mpisppy.convergers.norm_rho_converger``. An
example of use is shown in ``examples.farmer.farmer_cylinders.py``. This is
the original Gabe H. dynamic rho.


rho_setter
^^^^^^^^^^

Per variable rho values (mainly for PH) can be set using a function
that takes a scenario (a Pyomo ``ConcreteModel``) as its only
argument. The function returns a list of (id(vardata), rho)
tuples. The function name can be given the the ``vanilla.ph_hub``
constructor or in the hub dictionary under ``opt_kwargs`` as the
``rho_setter`` entry. (The function name is ultimately passed to the
``phabase`` constructor.)

There is an example of the function in the sizes example (``_rho_setter``).

SepRho
^^^^^^

Set per variable rho values using the "SEP" algorithm from

Progressive hedging innovations for a class of stochastic mixed-integer resource allocation problems
Jean-Paul Watson, David L. Woodruff, Compu Management Science, 2011
DOI 10.1007/s10287-010-0125-4

One can additional specify a multiplier on the computed value (default = 1.0).
If the cost coefficient on a non-anticipative variable is 0, the default rho value is used instead.

CoeffRho
^^^^^^^^

Set per variable rho values proportional to the cost coefficient on each non-anticipative variable,
with an optional multiplier (default = 1.0). If the coefficient is 0, the default rho value is used instead.

primal_dual_rho
^^^^^^^^^^^^^^^

Increase or decrease rho for every variable to keep primal and dual convergence balance. If
the primal residual is greater than ``update_threshold`` times the dual residual, then all
rhos are increased by the ``update_threshold``, and conversely all rhos are decreased if
the dual residual is greater than ``update_threshold`` time the primal residual. The user
can also specify a ``primal_bias`` (default 1.0) which will emphasize primal convergence
when greater than 1 and emphasize dual convergence if less than 1.

This extension is especially useful if the rhos provided by the user (or some other extension)
are believed to be "in balance", such that per-variable updates are not needed (and can sometimes
hinder algorithmic progress when different nonanticipative variables play similar roles in
the subproblem optimization problems).

wtracker_extension
^^^^^^^^^^^^^^^^^^

The wtracker_extension outputs a report about the convergence (or really, lack thereof) of
W values.
An example of its use is shown in ``examples.sizes.sizes_demo.py``


gradient_extension
^^^^^^^^^^^^^^^^^^
The gradient_extension sets gradient-based rho for PH.
An example of its use is shown in  ``examples.farmer.farmer_rho_demo.py``
There are options in ``cfg`` to control dynamic updates.

mult_rho_updater
^^^^^^^^^^^^^^^^

This extension does a simple multiplicative update of rho.

cross-scenario cuts
^^^^^^^^^^^^^^^^^^^
Two-stage models only. This extension adds cross scenario cuts as calculated
by the cross-scenario cut spoke. See the implementation paper for details.
An example of its use is shown in ``examples/farmer/cs_farmer.py``.


Distributed Subproblem Presolve
===============================
This functionality is available for all Hub and Spoke algorithms which inherit from
``SPBase``. It can be enabled by passing ``presolve=True`` into the constructor.

Leveraging the existing feasibility-based bounds tightening (FBBT) available in Pyomo, this
presolver will tighten the bounds on all variables, including the non-anticipative variables.
If the non-anticipative variables have different bounds, the bounds among the non-anticipative
variables will be synchronized to utilize the tightest available bound.

In its current state, the user might opt-in to presolve for two reasons:

1. For problems without relatively complete recourse, utilizing the tighter bounds on the
   non-anticipative variables and speed convergence and improve primal and dual bounds. In
   rare cases it might also detect infeasibility.

2. For problems where a "fixer" extension or spoke is used, determining tight bounds on the
   non-anticipative variables may improve the fixer's performance.

.. Note::
   Like many solvers, the presolver will convert infinite bounds to 1e+100.

.. Note::
   This capability requires the auto-persistent pyomo solver interface (APPSI) extensions
   for Pyomo to be built on your system. This can be achieved by running ``pyomo build-extensions``
   at the command line.

.. Note::
   The APPSI capability in Pyomo is under active development. As a result, the presolver
   may not work for all Pyomo models.


variable_probability
====================

This is experimental as of February 2021; use with caution.  The main use-case is
to allow zero-probability variables.

A function similar to ``rho_setter`` can be passed to the ``SPBase``
constructor via the ``PHBase`` construtor as the
``variable_probability`` argument to allow for per variable
probability specification. So it can be passed through by ``vanilla``
via ``ph_hub``. The function should return (vid, probability) pairs.
If the function needs arguments, pass them via
the ``SPBase`` option ``variable_probability_kwargs``

The variable probabilities impact the computation of
``xbars`` and ``W``.

.. Note::
   The only xhatter that is likely to work with variable probabilities is xhatxbar. The others
   are likely to execute without error messages but will not find good solutions.


Objective function considerations
---------------------------------

If variables with by-variable probability are in the objective function, it is
up to the scenario creator code to deal with it. This is not so difficult for
zero-probability variables.

zero-probability variables
--------------------------

When you
create the scenario, you probably want to fix zero probability variables and perhaps give
them a zero coefficient if they appear in the objective. Fixed
variables will not get a nonanticipativity constraint in bundles. If you
create the EF directly, you probably want to set
``nonant_for_fixed_vars`` to `False` in the call to ``create_EF``. If
you are not calling ``create_EF`` directly, but rather using the
``mpisppy.opt.ef.ExtensiveForm`` object, add ``nonant_for_fixed_vars``
to the dict passed as its ``options`` argument with the value
``False``.

.. Note::
   The ``W`` value for a zero-probability variable will be stay at zero.


Fixed variables may cause trouble if you are relying on the internal
PH convergence metric.

.. Note::
   You must declare variables to be in the nonant list even for those scenarios where they have
   zero probability if they are in other scenarios that share a scenario tree node at the variable's stage.


If some variables have zero probability in all scenarios, then you will need to set the option
``do_not_check_variable_probabilities`` to True in the options for ``spbase``. This will result in skipping the checks for
all variable probabilities! So you might want to set this to False to verify that the probabilities sum to one
only for the Vars you expect before setting it to True.

Scenario_lp_mps_writer
----------------------

This extension writes an lp file and an mps file with the model as well as a
json file with (a) list(s) of scenario tree node names and
nonanticaptive variables for each scenario before the iteration zero
solve of PH or APH. Note that for two-stage problems, all json files
will be the same. See ``mpisppy.generic_cylinders.py`` for an example
of use. In that program it is activated with the
``--scenario-lp-mps-writer`` option. Note that it
writes the files to the current working directory and for each scenario
the base name of the three files written is the scenario name.

Unless you know exactly why you need this, you probably don't.
