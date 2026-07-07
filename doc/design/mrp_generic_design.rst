========================================
Design Document: ``mrp_generic.py``
========================================

:Author: DLW
:Date: 2026-03-25
:Status: Draft

Overview
========

We want a generic, model-agnostic driver for **sequential sampling**
(the Multiple Replication Procedure, or MRP) analogous to what
``generic_cylinders.py`` provides for decomposition and MMW confidence
intervals.  The user should be able to point ``mrp_generic.py`` at any
model module (via ``--module-name``) and get a candidate solution
together with a confidence interval on its optimality gap, without
writing a custom ``xhat_generator`` or any model-specific sequential
sampling glue code.

Background
----------

Today, using sequential sampling requires the programmer to:

1. Write a model-specific ``xhat_generator`` function (see, e.g.,
   ``examples/farmer/CI/farmer_seqsampling.py:xhat_generator_farmer``
   or ``mpisppy/tests/examples/aircond.py:xhat_generator_aircond``).
2. Import ``SeqSampling`` (two-stage) or ``IndepScens_SeqSampling``
   (multi-stage) from ``mpisppy.confidence_intervals``.
3. Wire together the Config, the stopping criterion options (BM or
   BPL), and invoke ``sampler.run()``.

``generic_cylinders.py`` already eliminated equivalent boilerplate for
decomposition runs and for MMW confidence intervals.  This design aims
to do the same for sequential sampling.

Why Not Extend ``generic_cylinders.py``?
----------------------------------------

Sequential sampling has fundamentally different characteristics from
the decomposition workflow in ``generic_cylinders.py``:

- **Different termination criteria.**  ``generic_cylinders.py`` runs a
  single decomposition (PH/APH/FWPH/subgradient) that converges
  according to hub/spoke logic.  Sequential sampling iterates with
  increasing sample sizes until a *statistical* stopping criterion
  (BM or BPL) is satisfied.

- **No hub-and-spoke architecture.**  Sequential sampling does not
  launch cylinders or use ``WheelSpinner``.  Each iteration solves an
  EF (or possibly decomposition methods in the future) for a sample of
  scenarios, evaluates gap estimators, and decides whether to stop.

- **No lower bound production.**  The hub-and-spoke system is designed
  to produce and refine inner/outer bounds simultaneously.  MRP
  produces a *candidate solution* and a *confidence interval on the
  gap*, not a deterministic lower bound.

- **Different command-line interface.**  The user needs sequential
  sampling parameters (BM_h, BM_eps, BPL_eps, etc.) rather than
  cylinder/extension configuration.

For these reasons, a separate ``mrp_generic.py`` (living alongside
``generic_cylinders.py``) is the recommended approach.


Proposed Architecture
=====================

File Location and Name
----------------------

::

    mpisppy/mrp_generic.py          # top-level entry point (like generic_cylinders.py)
    mpisppy/generic/mrp.py          # core logic (like generic/mmw.py, generic/decomp.py)

The entry point ``mrp_generic.py`` will mirror the pattern of
``generic_cylinders.py``: parse ``--module-name``, load the module,
parse args, call the core logic.

Module Requirements
-------------------

The model module (e.g., ``farmer``) must already provide:

- ``scenario_creator(scenario_name, **kwargs)``
- ``scenario_names_creator(num_scens, start=None)``
- ``kw_creator(cfg)``
- ``inparser_adder(cfg)``
- ``scenario_denouement(...)``  (can be a no-op)

These are the same functions already required by ``generic_cylinders.py``
and by the existing sequential sampling code.  No new module functions
are required.

The Key Design Problem: A Generic ``xhat_generator``
----------------------------------------------------

The existing ``SeqSampling`` class requires an ``xhat_generator``
callback with this signature::

    def xhat_generator(scenario_names, solver_name=None,
                       solver_options=None, **kwargs) -> dict

This function must solve the approximate problem (usually an EF) for
the given scenarios and return a nonant cache (a dict mapping node
names to numpy arrays, e.g., ``{'ROOT': np.array([...])}``.

Today, every model that uses sequential sampling must provide its own
``xhat_generator``.  The key contribution of ``mrp_generic.py`` is
providing a **generic** one.

Proposed generic xhat_generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generic version will use ``Amalgamator`` (or directly
``ExtensiveForm``) in EF mode to solve the sample problem:

.. code-block:: python

    def generic_xhat_generator(scenario_names, solver_name=None,
                               solver_options=None, cfg=None,
                               module=None, start_seed=None,
                               branching_factors=None):
        """Model-agnostic xhat generator for sequential sampling.

        Solves the EF for the given scenario_names and returns
        the nonant cache.
        """
        num_scens = len(scenario_names)
        local_cfg = cfg()  # safe copy

        if branching_factors is not None:
            # multi-stage
            local_cfg.quick_assign("EF_mstage", bool, True)
        else:
            local_cfg.quick_assign("EF_2stage", bool, True)

        local_cfg.quick_assign("EF_solver_name", str, solver_name)
        local_cfg.quick_assign("EF_solver_options", dict, solver_options)
        local_cfg.quick_assign("num_scens", int, num_scens)
        local_cfg.quick_assign("_mpisppy_probability", float, 1/num_scens)
        if start_seed is not None:
            local_cfg.quick_assign("start_seed", int, start_seed)

        ama = amalgamator.from_module(module, local_cfg,
                                      use_command_line=False)
        ama.scenario_names = scenario_names
        ama.verbose = False
        ama.run()

        xhat = sputils.nonant_cache_from_ef(ama.ef)
        return xhat

This is essentially the pattern already used in the model-specific
generators (``farmer_seqsampling.py`` lines 58-79,
``seqsampling.py:xhat_generator_farmer``), but parameterized by the
module instead of being hard-coded.

**Open question:** Should we also support generating xhat via
decomposition (PH/cylinders) rather than EF?  This would be useful for
large problems where EF is intractable.  For now, we propose EF-only
(matching current ``SeqSampling`` capabilities) and flag this as future
work.

Two-Stage vs. Multi-Stage
--------------------------

The existing codebase supports both:

- ``SeqSampling`` (in ``seqsampling.py``) for two-stage problems
- ``IndepScens_SeqSampling`` (in ``multi_seqsampling.py``) for
  multi-stage problems using independent scenarios

``mrp_generic.py`` should handle both, selecting the appropriate class
based on the presence of ``--branching-factors``:

.. code-block:: python

    if cfg.get("branching_factors") is not None:
        # multi-stage
        sampler = IndepScens_SeqSampling(...)
    else:
        # two-stage
        sampler = SeqSampling(...)

Stopping Criterion Selection
------------------------------

The choice between BM and BPL stopping criteria will be a command-line
argument:

::

    --stopping-criterion BM   # Bayraksan and Morton (relative width)
    --stopping-criterion BPL  # Bayraksan and Pierre-Louis (fixed width)

Both sets of parameters (BM_h, BM_eps, etc. and BPL_eps, BPL_c0, etc.)
will be registered with the Config parser.  Parameters not relevant to
the chosen criterion will simply be ignored.


CLI Design
==========

The invocation will parallel ``generic_cylinders.py``:

::

    python -m mpisppy.mrp_generic --module-name farmer \
        --num-scens 3 \
        --solver-name cplex \
        --stopping-criterion BM \
        --BM-h 2.0 \
        --BM-q 1.3 \
        --confidence-level 0.95

For multi-stage::

    python -m mpisppy.mrp_generic --module-name aircond \
        --branching-factors "3 3 2" \
        --solver-name cplex \
        --stopping-criterion BM \
        --BM-h 0.55 \
        --BM-hprime 0.5 \
        --BM-eps 0.5 \
        --BM-eps-prime 0.4 \
        --BM-p 0.2 \
        --BM-q 1.2

Required arguments:

- ``--module-name`` (or ``--smps-dir`` / ``--mps-files-directory``)
- ``--solver-name``
- ``--stopping-criterion`` (BM or BPL)

Optional arguments (with defaults from ``confidence_config.py``):

- ``--confidence-level`` (default 0.95)
- ``--sample-size-ratio`` (default 1.0)
- ``--ArRP`` (default 1)
- BM parameters: ``--BM-h``, ``--BM-hprime``, ``--BM-eps``,
  ``--BM-eps-prime``, ``--BM-p``, ``--BM-q``
- BPL parameters: ``--BPL-eps``, ``--BPL-c0``, ``--BPL-n0min``

Output arguments:

- ``--solution-base-name`` (write xhat to file)
- ``--max-iterations`` (safety cap, default 200)


Proposed Code Structure
=======================

``mpisppy/generic/mrp.py``
--------------------------

Core logic module, analogous to ``generic/mmw.py``:

.. code-block:: python

    def mrp_args(cfg):
        """Register sequential sampling CLI arguments on cfg."""
        # stopping criterion choice
        cfg.add_to_config("stopping_criterion",
                          description="BM or BPL",
                          domain=str, default="BM")
        cfg.add_to_config("max_iterations",
                          description="Safety cap on sequential sampling iterations",
                          domain=int, default=200)
        # import existing config functions
        confidence_config.confidence_config(cfg)
        confidence_config.sequential_config(cfg)
        confidence_config.BM_config(cfg)
        confidence_config.BPL_config(cfg)

    def generic_xhat_generator(scenario_names, ...):
        """Model-agnostic xhat generator (see above)."""
        ...

    def do_mrp(module_fname, module, cfg):
        """Run sequential sampling and return results dict.

        Returns:
            dict with keys: T, Candidate_solution, CI
        """
        ...

``mpisppy/mrp_generic.py``
--------------------------

Top-level entry point:

.. code-block:: python

    if __name__ == "__main__":
        fname = model_fname()          # reuse from generic/parsing.py
        module = load_module(fname)
        cfg = parse_mrp_args(module)   # new parser, simpler than generic_cylinders
        result = do_mrp(fname, module, cfg)
        # output results

The ``parse_mrp_args`` function will be simpler than the
``generic_cylinders`` parser because it does not need cylinder, spoke,
extension, or bundling arguments.  It needs:

- Module loading args (``--module-name``)
- Model-specific args (via ``inparser_adder``)
- Solver args
- Sequential sampling args (via ``mrp_args``)
- Solution output args


Design Decisions to Make
========================

1. **EF and decomposition for xhat generation**

   *Decision:* Support both from the start via ``--xhat-method``:

   - ``--xhat-method EF`` (default): solve each sample via EF using
     ``Amalgamator``.  This is the traditional approach matching the
     existing ``SeqSampling`` examples.
   - ``--xhat-method cylinders``: spin up a ``WheelSpinner`` for each
     sample.  Heavier, but enables problems too large for EF.

   The cylinder path uses ``do_decomp`` from ``generic/decomp.py``
   and extracts xhat via the same temp-file approach as
   ``generic/mmw.py``.  When ``--xhat-method=cylinders`` is selected,
   ``parse_mrp_args`` registers the decomposition CLI args (PH, APH,
   spokes, etc.) so the user can configure cylinders normally.

2. **Solver specification**

   The existing ``SeqSampling`` uses ``solver_spec.solver_specification``
   to extract the solver, while ``generic_cylinders`` uses
   ``cfg.solver_name`` directly.  We should use the same pattern as
   ``generic_cylinders`` (``--solver-name``) and translate internally.

   *Recommendation:* Accept ``--solver-name`` and set
   ``EF_solver_name`` internally, as ``generic/mmw.py`` already does.

3. **Integration with generic_cylinders**

   Should it be possible to invoke MRP from within a ``generic_cylinders``
   run (like MMW is invoked after decomposition)?  This is conceptually
   different: MMW evaluates a *given* xhat, while MRP *produces* an xhat.
   Running MRP after decomposition would be redundant.

   *Recommendation:* Keep them separate.  ``mrp_generic.py`` is a
   standalone entry point.  A user who wants to compare MRP against
   decomposition+MMW runs them as separate invocations.

4. **Stochastic vs. deterministic sample sizes**

   The ``stochastic_sampling`` flag in ``SeqSampling`` enables §5 of
   [BPL2012].  This should be exposed via CLI, probably keyed off
   ``--BPL-n0min`` being nonzero (matching ``farmer_seqsampling.py``
   behavior).

   *Recommendation:* ``stochastic_sampling = (BPL_n0min != 0)`` when
   the stopping criterion is BPL.

5. **Module name string propagation through the CI stack**

   The entire confidence interval call chain is wired around passing
   the model as an **importable string** (e.g., ``"farmer"`` or
   ``"mpisppy.tests.examples.aircond"``), not as a module object.
   The string gets re-imported via ``importlib.import_module()``
   independently at multiple levels:

   a. ``SeqSampling.__init__`` (``seqsampling.py:166``) does
      ``importlib.import_module(refmodel)`` and stores both
      ``self.refmodel`` (the module object, used locally for
      ``scenario_names_creator`` etc.) and ``self.refmodelname``
      (the string, passed downstream).

   b. ``SeqSampling.run()`` passes ``self.refmodelname`` to
      ``ciutils.gap_estimators()`` (lines 418, 497).

   c. ``ciutils.gap_estimators()`` (``ciutils.py:279``) does
      ``importlib.import_module(mname)`` *again* to call
      ``kw_creator``.  It also passes the string to
      ``sample_tree.SampleSubtree()`` and
      ``amalgamator.from_module()``.

   d. ``SampleSubtree.__init__`` (``sample_tree.py:63``) does
      ``importlib.import_module(mname)`` *yet again*.

   e. ``walking_tree_xhats`` passes the string further into more
      ``SampleSubtree`` and ``amalgamator.from_module`` calls.

   So in each iteration of the sequential sampling loop, the module
   is re-imported 3-4 times.  This is normally harmless (Python
   caches modules in ``sys.modules``), but it means every function
   in the chain **requires a string**, not a module object.

   **Implication for mrp_generic:**  In ``generic_cylinders.py``,
   the module is loaded via ``load_module(fname)`` which does
   ``sys.path.append(dpath)`` followed by
   ``importlib.import_module(basename)``.  The basename (e.g.,
   ``"farmer"``) becomes importable because its directory was added
   to ``sys.path``.  So we can pass that basename string into
   ``SeqSampling`` and it will work — as long as we strip the path
   component first.

   ``generic/mmw.py`` already handles this translation at line 52::

       module_fname = os.path.basename(module_fname)

   We need the same trick for MRP.

   **Subtle risk:** If the user runs
   ``--module-name /some/path/to/mymodel``, ``load_module`` makes
   ``"mymodel"`` importable, but the CI internals re-import it
   independently each time.  They'll get the *same* cached module
   object (via ``sys.modules``), so this is safe — unless the module
   has mutable global state that gets modified between imports.
   This is unlikely but worth noting.

   *Recommendation for Phase 1:*  Do what ``generic/mmw.py`` does —
   pass ``os.path.basename(module_fname)`` as the string to
   ``SeqSampling``.

   *Recommendation for a follow-up PR:*  Refactor the CI stack so
   that ``SeqSampling``, ``gap_estimators``, ``SampleSubtree``, and
   ``walking_tree_xhats`` accept a module object instead of a
   string.  This eliminates the repeated ``importlib.import_module``
   calls and makes the API cleaner.  This is a larger change
   touching ``ciutils.py``, ``sample_tree.py``,
   ``seqsampling.py``, ``multi_seqsampling.py``, and
   ``mmw_ci.py``, so it should be a separate PR.

   Also note: ``seqsampling.py`` currently has a bare
   ``print("\nTBD: check seqsampling for start vs start_seed")``
   at module level (line 32) that fires on every import.  This
   should be cleaned up regardless.

6. **Output format**

   *Recommendation:* Print the results dict (T, CI, candidate solution)
   to stdout.  Optionally write xhat to a file via
   ``--solution-base-name`` (as ``.npy`` and ``.csv``).


Implementation Plan
===================

Phase 1: Core functionality (DONE)
-----------------------------------

1. Created ``mpisppy/generic/mrp.py`` with:

   - ``mrp_args(cfg)`` — registers CLI args
   - ``_ef_xhat_generator(...)`` — model-agnostic EF-based xhat generator
   - ``_cylinder_xhat_generator(...)`` — xhat generator via decomposition
   - ``do_mrp(module_fname, module, cfg)`` — orchestrates sequential sampling

2. Created ``mpisppy/mrp_generic.py`` with:

   - CLI entry point using ``model_fname()`` / ``load_module()`` from
     ``generic/parsing.py``
   - ``parse_mrp_args(module)`` — builds Config with appropriate args
     (includes decomposition args for ``--xhat-method=cylinders``)
   - ``__main__`` block with result printing and optional .npy output

3. Updated ``mpisppy/generic/__init__.py`` to export ``do_mrp``.

Phase 2: Testing
----------------

4. Add a test that runs ``mrp_generic`` on the farmer model with known
   parameters and verifies the output structure (T, CI, candidate
   solution).  This can be modeled on the existing farmer sequential
   sampling example.

5. Add a multi-stage test using the aircond model.

6. Test ``--xhat-method=cylinders`` path with a small farmer problem.

Phase 3: Documentation and examples
------------------------------------

7. Add ``doc/src/mrp_generic.rst`` documentation.

8. Update ``doc/src/seqsamp.rst`` to reference the generic driver.

9. Add example bash scripts analogous to
   ``examples/farmer/CI/farmer_sequential.bash``.

Phase 4: Future enhancements (separate PRs)
--------------------------------------------

10. Refactor ``SeqSampling`` to accept a module object directly
    (rather than a string for ``importlib``).

11. Investigate whether ``solving_type`` could support cylinder-based
    solving for the gap estimation step (not just xhat generation).
