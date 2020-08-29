.. _scenario_creator:

`scenario_creator` function
===========================

This function is required, but can have any name, and its first
argment must be the scenario name. The other two arguments are
optional. The`cb_data` option specifies data that is passed through
from the calling program all the way to the scenario creator function
(it is callded `cb_data`). This can be of any type, and might give,
e.g., a data directory name or probability distribution information.
The function returns an instantiated model for the instance. I.e.,
it either creates a `ConcreteModel` or else it creates and instantiates
an `AbstractModel`.

The `scenario_creator` function somehow needs to create a list of non-leaf tree node
objects that are constructed by calling `scenario_tree.ScenarioNode`
which is not very hard for two stage problems, because there is only
one non-leaf node and it must be called "ROOT".  If there are other
nodes, their names, although strings, must either be a unique integer
or end in a unique integer (e.g. "1" or "Node1", etc.) The node
constructor takes as arguments:

* name,
* conditional probability,
* stage number,
* stage cost expression,
* list of scenario names at the node (optional and not used)
* list of Vars subject to non-anticipativity at the node (I think slicing is allowed)
* the concrete model instance.

This node list must be attached to the scenario model instance under
the name `model._PySPnode_list`.
  
In the `general_farmer.py` example, the `scenario_creator` function is
called `pysp2_callback` and in this example, the scenario name is presumed
to be of the form "scen" with a trailing number. The trailing number is
used in a problem-specific way to create a "farmer" problem instance. The
concrete model instance is created.

We can get the probability from scenario creator attached to scenario
as PySP_prob, but if you don't attach it, they are assumed to be
equally likely. 
