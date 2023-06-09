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
with ``mpi-sspy`` and they provide examples that can be used for the
creation of more. Extensions can be found in ``mpisppy.extensions``.

Multiple Extensions
-------------------

To employ multiple PH extensions, use ``mpisppy.extensions.extension import MultiExtension``
that allows you to give a list of extensions that will fire in order
at each callout point. See, e.g. ``examples.sizes.sizes_demo.py`` for an
example of use.


PH extensions
-------------

Some of these can be used with other hubs. An extension object can be
passed to the PH constructor and it is assumed to have methods defined
for all the callout points in PH (so all of the examples do).  If you
want to use more than one extension, define a main extension that has
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
is an example of its use in ``examples.sizes.sizes_demo.py``

fixer.py
^^^^^^^^

This extension provides methods for fixing variables (usually integers) for
which all scenarios have agreed for some number of iterations. There
is an example of its use in ``examples.sizes.sizes_demo.py`` also
in ``examples.sizes.uc_ama.py``. The ``uc_ama`` example illustrates
that when ``amgalgamator`` is used ``"id_fix_list_fct"`` needs
to be on the ``Config`` object so the amalgamator can find it.

xhat
^^^^

Most of the xhat methods can be used as an extension instead of being used
as a spoke, when that is desired (e.g. for serial applications).

WXBarWriter and WXBarReader
^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is an extension to write xbar and W values and another to read them.
An example of their use is shown in ``examples.sizes.sizes_demo.py``

norm_rho_updater
^^^^^^^^^^^^^^^^

This extension adjust rho dynamically. The code is in ``mpisppy.extensions.norm_rho_updater.py``
and there is an accompanying converger in ``mpisppy.convergers.norm_rho_converger``. An
example of use is shown in ``examples.farmer.farmer_cylinders.py``.


rho_setter
==========

Per variable rho values (mainly for PH) can be set using a function
that takes a scenario (a Pyomo ``ConcreteModel``) as its only
argument. The function returns a list of (id(vardata), rho)
tuples. The function name can be given the the ``vanilla.ph_hub``
constructor or in the hub dictionary under ``opt_kwargs`` as the
``rho_setter`` entry. (The function name is ultimately passed to the
``phabase`` constructor.)

There is an example of the function in the sizes example (``_rho_setter``).


wtracker_extension
==================

The wtracker_extension outputs a report about the convergence (or really, lack thereof) of
W values.
An example of its use is shown in ``examples.sizes.sizes_demo.py``


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


gradient_extension
==================
The gradient_extension sets adaptative gradient-based rho for PH.
An example of its use is shown in  ``examples.farmer.farmer_rho_demo.py``


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
