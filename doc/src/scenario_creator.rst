.. _scenario_creator:

`scenario_creator` function
===========================

This function instantiates models for scenarios and usually attaches
some information about the scenario tree. It is required, but can have
any name, and its first argment must be the scenario name. The other
two arguments are optional. The `scenario_creator_kwargs` option specifies data that is
passed through from the calling program.
`scenario_creator_kwargs` must be a dictionary, and might give, e.g., a data
directory name or probability distribution information.  The
`scenario_creator` function returns an instantiated model for the
instance. I.e., it either creates a `ConcreteModel` or else it creates
and instantiates an `AbstractModel`.

The `scenario_creator` function somehow needs to create a list of
non-leaf tree node objects that are constructed by calling
`scenario_tree.ScenarioNode` which is not very hard for two stage
problems, because there is only one non-leaf node and it must be
called "ROOT".  If there are other scenario tree nodes, their names,
although strings, must indicates their position in the tree, 
like "ROOT_3_0_1". A given non-root node, which is the child number `k` of
a node with name `parentname`, should be named `parentname_k`.
The node constructor takes as
arguments:

* name,
* conditional probability,
* stage number,
* stage cost expression,
* list of scenario names at the node (optional and not used)
* list of Vars subject to non-anticipativity at the node (I think slicing is allowed)
* the concrete model instance.

This node list must be attached to the scenario model instance under
the name `model._mpisppy_node_list`.
  
In the `farmer.py` example, the `scenario_creator` function is called
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
   
   # Add all index of model.x using wild cards
   sputils.attach_root_node(model, model.FirstStageCost, [model.x[:,:], ])

The scenario probability should be attached by `scenario_creator` as
``_mpisppy_probability``. However, if you don't attach it, the scenarios are
assumed to be equally likely.

EF Supplement List
------------------

The function ``attach_root_node`` takes an optional argument ``nonant_ef_suppl_list`` (that is passed through to the ``ScenarioNode`` constructor). This is a list similar to the nonanticipate Var list. These variables will not be given
multipliers by algorithms such as PH, but will be given non-anticipativity
constraints when an EF is formed, either to solve the EF or when bundles are
formed. For some problems, with the appropriate solver, adding redundant nonanticipativity constraints
for auxilliary variables to the bundle/EF will result in a (much) smaller pre-solved model.



