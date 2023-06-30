.. _Internals:

Internals
=========

In this section we provide some notes concerning the code intended to help
developers or maintainers get started.

``nonant_vardata_list``
-----------------------

An important internal structure is ``nonant_vardata_list``, which contains references to the Pyomo vardata objects
for the nonanticipative variables. For indexed Vars, there is one entry for each index. This list enables querying or
setting the values of the nonanticipative variables in an abstract way.

One way or the other, the scenario creator function
needs to attached to each scenario model a list of `objects for non-leaf scenario tree nodes (the list will have only one member for two-stage problems)
called ``_mpisppy_node_list``.
Each node object contains the ``nonant_vardata_list`` for the scenaio at that scenario tree node. The list indexes are not meaningful, but the
lists are assumed to be in the same order for every scenario at the tree node.

Some uses
^^^^^^^^^

A simple use is found in ``sputils.py``

.. code-block:: python

   def first_stage_nonant_npy_serializer(file_name, scenario, bundling):
        # write just the nonants for ROOT in an npy file (e.g. for Conf Int)
       root = scenario._mpisppy_node_list[0]
       assert root.name == "ROOT"
       root_nonants = np.fromiter((pyo.value(var) for var in root.nonant_vardata_list), float)
       np.save(file_name, root_nonants)

Another simple use can be found in `xhat_eval.py`:


.. code-block:: python
		
   def fix_nonants_upto_stage(self,t,cache):


Standard Construction
^^^^^^^^^^^^^^^^^^^^^

If you are going to use the ``vardatalist`` you don't really need to know how it is constructed, but
we will illustrate one common way here.
Many modelers/users have a call to ``attach_root_node`` in their scenario creator function (e.g., see the scenario scenario
creator in the farmer example), which is a utility
for two-stage problems only. Here is the function signature:

.. code-block:: python

    def attach_root_node(model, firstobj, varlist, nonant_ef_suppl_list=None):

inside the function, there is this call

.. code-block:: python

    model._mpisppy_node_list = [
        scenario_tree.ScenarioNode("ROOT", 1.0, 1, firstobj, varlist, model,
                                   nonant_ef_suppl_list = nonant_ef_suppl_list)
    ]

that attaches a node list with one ROOT node to the model object of the scenario. The ``ScenarioNode`` constructor
creates the ``nonant_vardata_list`` with this call

.. code-block:: python

   self.nonant_vardata_list = build_vardatalist(self,
                                                scen_model,
                                                self.nonant_list)

The ``build_vardatalist`` 

nonant_ef_suppl_list
^^^^^^^^^^^^^^^^^^^^

The ``nonant_ef_suppl_list`` supplied (optinally) to the ``ScenarioNode`` constructor
sets up a nonant var data list for Vars whose nananticipative is enforced only in extensive
forms (which includes bundles). This can be useful for supplemental variables whose values
are implied by other nonanticipative variables (e.g. indicator variables).
