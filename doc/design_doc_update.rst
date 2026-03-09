Documentation Restructuring Design
====================================

This document outlines the planned reorganization of the mpi-sppy
RST documentation. The overarching goal is to make ``generic_cylinders.py``
the clear, primary entry point and to demote custom drivers and legacy
workflows (PySP, Amalgamator, old ``*_cylinders.py`` scripts) to
appendix/developer material.

Guiding Principles
------------------

1. **generic_cylinders is the front door.** New users should never need to
   write their own driver. The docs should make this obvious from the
   first page.

2. **Layered depth.** Quick start -> model file howto -> choosing
   algorithms/spokes -> advanced topics -> developer internals.

3. **Remove or downgrade stale content.** PySP conversion, Amalgamator,
   hand-rolled drivers, and the "alpha-release" caveat on
   generic_cylinders are all outdated.

4. **Add missing topics.** The Config system, rho-setting strategies,
   and multistage-specific guidance are under-documented.


Proposed Table of Contents
--------------------------

Below is the new ``index.rst`` toctree, with notes on each entry.

::

  Part I -- Getting Started
  ~~~~~~~~~~~~~~~~~~~~~~~~~
  quick_start.rst          (REVISE - see below)
  install_mpi.rst          (minor updates)

  Part II -- Building Your Model
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  scenario_creator.rst     (REVISE - tighten, add multistage guidance)
  helper_functions.rst     (REVISE - clarify required vs optional functions)

  Part III -- Running with generic_cylinders
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  generic_cylinders.rst    (REVISE HEAVILY - this becomes the centerpiece)
  examples.rst             (REVISE - rewrite around generic_cylinders usage)
  ef.rst                   (REVISE - frame as "the --EF flag" not a separate API)

  Part IV -- Algorithms and Cylinders
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  overview.rst             (REVISE - keep architecture explanation, update language)
  hubs.rst                 (REVISE - describe from command-line flag perspective)
  spokes.rst               (REVISE - describe from command-line flag perspective)
  extensions.rst           (REVISE - which ones work via generic_cylinders flags)
  rho_setting.rst          (NEW - consolidate grad_rho, sep_rho, coeff_rho,
                             sensi_rho, norm_rho, primal_dual_rho into one page)

  Part V -- Solutions and Confidence Intervals
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  output_solutions.rst     (REVISE - emphasize --solution-base-name)
  access_solutions.rst     (REVISE - minor)
  confidence_intervals.rst (REVISE - show MMW via generic_cylinders flags)
  zhat.rst                 (minor updates)
  seqsamp.rst              (minor updates)

  Part VI -- Advanced Topics
  ~~~~~~~~~~~~~~~~~~~~~~~~~~
  properbundles.rst        (REVISE - minor, already decent)
  agnostic.rst             (REVISE - minor)
  admmWrapper.rst          (keep)
  stoch_admmWrapper.rst    (keep)
  config.rst               (NEW - document the Config system, popular_args, etc.)
  aph.rst                  (minor updates)
  nompi4py.rst             (keep, short)
  secretmenu.rst           (keep)
  w_rho.rst                (fold into rho_setting.rst or keep as stub)

  Part VII -- For Developers and Contributors
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  drivers.rst              (REVISE - retitle "Writing Custom Drivers (Advanced)",
                             add clear note that this is rarely needed)
  internals.rst            (keep)
  contributors.rst         (keep)
  amalgamator.rst          (REVISE - mark as legacy/deprecated)
  pysp.rst                 (REVISE - mark as legacy)

  References
  ~~~~~~~~~~
  api.rst
  refs.rst


Detailed Changes Per File
-------------------------

quick_start.rst
^^^^^^^^^^^^^^^

- Remove PySP section entirely (move to pysp.rst appendix).
- Lead with: install, verify, run farmer via generic_cylinders.
- Show the two main modes: ``--EF`` (no MPI needed) and decomposition
  (``mpiexec``).
- Add a "what you need to provide" summary: a module with
  ``scenario_creator``, ``scenario_names_creator``, ``kw_creator``,
  ``inparser_adder``, ``scenario_denouement``.
- Keep the "researchers who want to compare" section but update commands.

generic_cylinders.rst
^^^^^^^^^^^^^^^^^^^^^

This is the most important file to rewrite. Current version says
"alpha-release" -- remove that.

Proposed structure:

1. **Introduction** -- this is the recommended way to use mpi-sppy.
2. **Your model file (module)** -- what functions it must contain,
   with cross-refs to scenario_creator.rst and helper_functions.rst.
3. **Solving the EF** -- ``python generic_cylinders.py --module-name foo --EF ...``
4. **Running PH with spokes** -- ``mpiexec -np N python -m mpi4py
   generic_cylinders.py --module-name foo --lagrangian --xhatshuffle ...``
5. **Choosing a hub algorithm** -- ``--APH``, ``--subgradient-hub``,
   ``--fwph-hub``, ``--ph-primal-hub`` (default is PH).
6. **Choosing spokes** -- table of flags: ``--lagrangian``,
   ``--xhatshuffle``, ``--xhatxbar``, ``--fwph``, ``--subgradient``,
   ``--ph-dual``, ``--relaxed-ph``, ``--reduced-costs``.
7. **Rho settings** -- overview with cross-ref to rho_setting.rst.
8. **Extensions via command line** -- ``--fixer``, ``--mipgaps-json``,
   ``--user-defined-extensions``.
9. **Solution output** -- ``--solution-base-name``, custom writers.
10. **MMW confidence intervals** -- ``--mmw-num-batches``, etc.
11. **Pickling scenarios and bundles** -- existing content, cleaned up.
12. **Advanced: hub_and_spoke_dict_callback** -- existing content.
13. **Advanced: using a class in the module** -- existing content.
14. **Config files** -- ``--config-file``.
15. **Presolve (FBBT/OBBT)** -- existing content.

examples.rst
^^^^^^^^^^^^

- Rewrite the farmer walkthrough to use generic_cylinders throughout.
- For each example, show the generic_cylinders command (not a custom
  ``*_cylinders.py`` script).
- Move the ``*_cylinders.py`` references to a "historical note" box.

drivers.rst
^^^^^^^^^^^

- Retitle: "Writing Custom Drivers (Advanced/Historical)"
- Add a prominent note at the top: "Most users should use
  ``generic_cylinders.py``. This section is for developers who need
  features not yet exposed through generic_cylinders or who want to
  understand the internal architecture."
- Keep the existing content mostly as-is for developer reference.

overview.rst
^^^^^^^^^^^^

- Update the Roles section: mention that generic_cylinders has reduced
  the need for a separate "Developer" role in many cases.
- Remove or soften "Neither this document, nor mpi-sppy are written
  with the intention that they will be employed directly by end-users."
  With generic_cylinders, modelers *can* be end-users.

rho_setting.rst (NEW)
^^^^^^^^^^^^^^^^^^^^^

Consolidate rho documentation into one page:

- ``--default-rho``: the baseline
- ``_rho_setter`` function in the module
- ``--sep-rho``: separation-based
- ``--coeff-rho``: coefficient-based
- ``--sensi-rho``: sensitivity-based
- ``--reduced-costs-rho``: reduced-costs-based
- ``--grad-rho``: gradient-based (absorb current grad_rho.rst)
- ``--use-norm-rho-updater``: norm-based adaptive
- ``--use-primal-dual-rho-updater``: primal-dual adaptive
- ``--dynamic-rho``: dynamic updates
- Guidance on when to use which strategy.

Keep grad_rho.rst as a redirect/stub pointing to rho_setting.rst,
or remove it from toctree.

config.rst (NEW)
^^^^^^^^^^^^^^^^

- Explain the ``Config`` class (extends Pyomo ConfigDict).
- Show how ``popular_args()``, ``ph_args()``, etc. work.
- Document ``parse_command_line()``, ``add_to_config()``,
  ``quick_assign()``.
- Explain ``--config-file`` usage.
- Note: this is mostly needed by advanced users and developers.

helper_functions.rst
^^^^^^^^^^^^^^^^^^^^

- Clarify which functions are required vs optional.
- Required: ``scenario_creator``, ``scenario_names_creator``,
  ``kw_creator``, ``inparser_adder``, ``scenario_denouement``.
- Optional: ``_rho_setter``, ``id_fix_list_fct``,
  ``hub_and_spoke_dict_callback``, ``custom_writer``,
  ``get_mpisppy_helper_object``.

hubs.rst / spokes.rst
^^^^^^^^^^^^^^^^^^^^^

- Add a column or note for each hub/spoke showing the
  generic_cylinders command-line flag that enables it.
- Keep the algorithmic descriptions.

ef.rst
^^^^^^

- Reframe: "Solving the Extensive Form" with primary path being
  ``generic_cylinders.py --EF``.
- Keep the ``ExtensiveForm`` class API docs for developers.
- Keep ``create_EF`` for the no-MPI case.

amalgamator.rst
^^^^^^^^^^^^^^^

- Add deprecation notice: "The Amalgamator is a legacy wrapper.
  New users should use ``generic_cylinders.py`` instead."

pysp.rst
^^^^^^^^

- Add legacy notice: "PySP support is maintained for backward
  compatibility. New projects should write a ``scenario_creator``
  function directly."

confidence_intervals.rst
^^^^^^^^^^^^^^^^^^^^^^^^

- Show how to run MMW directly from generic_cylinders flags.
- Keep standalone ``mmw_conf`` usage as alternative.

extensions.rst
^^^^^^^^^^^^^^

- Add a section or table showing which extensions can be activated
  via generic_cylinders flags vs which require custom driver code.

Files to Remove from toctree (or merge)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``w_rho.rst`` -- fold content into ``rho_setting.rst``
- ``grad_rho.rst`` -- fold content into ``rho_setting.rst``

Implementation Order
--------------------

Suggested order of implementation:

1. Create new ``index.rst`` with the reorganized toctree (using
   parts/sections as shown above).
2. Create ``rho_setting.rst`` and ``config.rst`` (new files).
3. Revise ``generic_cylinders.rst`` (highest impact).
4. Revise ``quick_start.rst``.
5. Revise ``overview.rst``, ``drivers.rst``, ``examples.rst``.
6. Revise ``helper_functions.rst``, ``ef.rst``.
7. Add legacy notices to ``amalgamator.rst``, ``pysp.rst``.
8. Update ``hubs.rst``, ``spokes.rst``, ``extensions.rst`` with
   command-line flag references.
9. Minor updates to remaining files.
10. Build docs and fix any cross-reference issues.
