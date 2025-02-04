.. _scenario_creator:

`scenario_creator` function
===========================

This function instantiates models for scenarios and usually attaches
some information about the scenario tree. It is required, but can have
any name, and its first argument must be the scenario name. The other
two arguments are optional. The `scenario_creator_kwargs` option specifies data that is
passed through from the calling program.
`scenario_creator_kwargs` must be a dictionary, and might give, e.g., a data
directory name or probability distribution information.  The
`scenario_creator` function returns an instantiated model for the
instance. I.e., it either creates a `ConcreteModel` or else it creates
and instantiates an `AbstractModel`.

Scenario Tree Information
-------------------------

The `scenario_creator` function somehow needs to create a list of
non-leaf tree node objects that are constructed by calling
`scenario_tree.ScenarioNode` which is not very hard for two stage
problems, because there is only one non-leaf node and it must be
called "ROOT". There is a helper function ``mpisppy.sputils.attach_root_node``

Multi-stage problems are discussed below.

Examples
--------

In the `farmer.py` example, the `scenario_creator` function calls
`pysp2_callback` and in this example, the scenario name is presumed to
be of the form "scen" with a trailing number. The trailing number is
used in a problem-specific way to create a "farmer" problem
instance. The concrete model instance is created.

The scenario creator
function in ``examples.netdes.netdes.py`` is very simple and
illustrates use of the utility function
(``mpisppy.utils.sputils.attach_root_node``) that attaches the node
list for you.

Node list entries can be entered indididually, by adding an entire
variable implicitly including all index values, and/or by using wildcards. This is
illustrated in the netdes example:

::
   
   # Add all indexes of model.x
   sputils.attach_root_node(model, model.FirstStageCost, [model.x, ])

::
   
   # Add all indexes of model.x using wild cards
   sputils.attach_root_node(model, model.FirstStageCost, [model.x[:,:], ])

Scenario Probability
--------------------

The scenario probability should be attached by `scenario_creator` as
``_mpisppy_probability``. However, if you don't attach it, the scenarios are
assumed to be equally likely. If the scenarios are equally likely, you
can avoid a warning by assigning the string "uniform" to the
``_mpisppy_probability`` attribute of the scenario model.

EF Supplement List
------------------

The function ``attach_root_node`` takes an optional argument ``nonant_ef_suppl_list`` (that is passed through to the ``ScenarioNode`` constructor). This is a list similar to the nonanticipate Var list. These variables will not be given
multipliers by algorithms such as PH, but will be given non-anticipativity
constraints when an EF is formed, either to solve the EF or when bundles are
formed. For some problems, with the appropriate solver, adding redundant nonanticipativity constraints
for auxiliary variables to the bundle/EF will result in a (much) smaller pre-solved model.

Surrogate Nonant List
---------------------

The function ``attach_root_node`` takes an additional optional argument ``surrogate_nonant_list`` (that is also passed through to the ``ScenarioNode`` constructor).
This list is similar to the nonanticipative Var list.
These variable *will not* be used when forming nonanticipativity constraints in the EF and *will not* be fixed in incubment finders *nor* in fixing heuristics.
However, these variables *will* be given multipliers by algorithms such as PH.
The nonanticipativity of these variables should be implied by the nonanticipativity of the variables in the Node list.
These variables are sometimes useful to help iterative algorithms converge faster by capturing hierarchical model decisions.
See ``examples/sslp/sslp.py`` for an example, where the single surrogate nonanticipative variable captures the total number of installed servers.

Multi-stage
-----------

When there are scenario tree nodes other than root, their names,
although strings, must indicate their position in the tree, 
like "ROOT_3_0_1". A given non-root node, which is the child number `k` of
a node with name `parentname`, should be named `parentname_k`.
The node constructor, ``scenario_tree.ScenarioNode`` takes as
arguments:

* name,
* conditional probability,
* stage number,
* stage cost expression,
* list of scenario names at the node (optional and not used)
* list of Vars subject to non-anticipativity at the node (I think slicing is allowed)
* the concrete model instance.

This node list must be attached to the scenario model instance under
the name `model._mpisppy_node_list`. See, e.g., the AirCond example.

Speed-up
========

If you are using a Pyomo model, you might be able to get a big speed
up by pickling your base model. Your `scenario_creator` function can
load the pickle file quickly, then make the modifications to the base
model according to the scenario. You should probably use the `dill`
package if you do this.  Note that if you run the same large base
model every time you run, this might be a very significant speed-up. (To be
extra clear: this suggestion is not related to pickling scenarios, but
rather to pickling a common base model).

