.. _mutable_probability:

Mutable Scenario Probabilities
==============================

By default, each scenario's probability is folded into the Extensive Form
(EF) objective as a floating-point constant when the model is built. Changing
a probability therefore requires rebuilding the objective and, for a
persistent solver, re-loading the instance.

The ``mutable_probability`` option (issue #797) makes the probabilities
updatable in place. When it is set, each scenario's probability is stored as a
mutable Pyomo ``Param`` in the objective instead of a constant, so the
probability vector can be changed and the (persistent) solver re-solved
cheaply, without rebuilding the model. This is useful for probability
sensitivity studies and rolling-horizon loops that keep a fixed scenario set
and only re-weight it between solves.

Requirements and scope
-----------------------

- The supplied probabilities must sum to 1 (option "B" in the design). There
  is no re-normalization: a vector that does not sum to 1 (within ``1e-9``) is
  rejected. This keeps the objective a plain probability-weighted sum with no
  division node.
- ``mutable_probability`` is a full-EF feature and is **not** supported for
  scenario bundles (which rely on the normalization divisor). Requesting it
  for a bundle raises an error.
- Updates are transactional: if a call is rejected, no probability is changed.

Extensive Form
--------------

Build the EF with ``mutable_probability=True``, then call
``set_scenario_probabilities`` with a mapping and re-solve. Pass
``reuse_instance=True`` after the first solve so a persistent solver keeps its
loaded instance and only the objective coefficients are re-pushed:

.. code-block:: python

   from mpisppy.opt.ef import ExtensiveForm

   ef = ExtensiveForm(
       options={"solver": "gurobi_persistent"},
       all_scenario_names=scenario_names,
       scenario_creator=scenario_creator,
       scenario_creator_kwargs=scenario_creator_kwargs,
       mutable_probability=True,
   )

   for i, prob_map in enumerate(probability_vectors):
       ef.set_scenario_probabilities(prob_map)   # must sum to 1
       ef.solve_extensive_form(reuse_instance=(i > 0))
       print(ef.get_objective_value(), ef.get_root_solution())

A partial mapping is allowed: scenarios omitted from ``prob_map`` keep their
current probability, as long as the resulting full vector still sums to 1.

Persistent solvers, including the APPSI / ``pyomo.contrib.solver`` interface
(e.g. ``appsi_highs``, the solver in the issue) are recognized via
``sputils.has_persistent_solve_api`` so the ``reuse_instance`` path is taken.
Legacy persistent solvers (e.g. ``gurobi_persistent``) require the objective
to be re-pushed, which ``set_scenario_probabilities`` handles; APPSI solvers
auto-track the change on the next solve.

A complete, runnable example is ``examples/farmer/farmer_prob_sensitivity.py``,
which sweeps the weight on one farmer scenario and reports how the optimal
first-stage planting decision responds.

Progressive Hedging and other decomposition
--------------------------------------------

The PH family does not bake probabilities into a Pyomo objective; it weights
``xbar``/``W``/``rho`` numerically through ``prob_coeff``, read fresh each
iteration. ``SPBase.set_scenario_probabilities`` updates
``_mpisppy_probability`` on the scenarios and forces ``prob_coeff`` to be
recomputed so the next iteration uses the new weights:

.. code-block:: python

   ph.set_scenario_probabilities(prob_map, reset_ph_duals=True)

``reset_ph_duals=True`` (the default) zeroes the PH multipliers ``W``. This
matters when re-solving from a *converged* PH solution: there, every scenario
sits at the same nonanticipative point, so the probability-weighted ``xbar``
is independent of the weights and PH would otherwise report immediate (false)
convergence at the old solution. Zeroing ``W`` breaks that consensus so PH
re-converges for the new probabilities.

.. note::

   ``SPBase.set_scenario_probabilities`` currently supports two-stage
   problems. Multistage node probabilities (``ScenarioNode.cond_prob``) are a
   later phase and raise ``NotImplementedError``.

API
---

.. automethod:: mpisppy.opt.ef.ExtensiveForm.set_scenario_probabilities
   :noindex:

.. automethod:: mpisppy.spbase.SPBase.set_scenario_probabilities
   :noindex:
