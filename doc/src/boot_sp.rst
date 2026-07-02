.. _boot_sp:

Bootstrap Confidence Intervals
==============================

The ``mpisppy.confidence_intervals.bootsp`` subpackage provides bootstrap
and bagging confidence intervals for the optimality gap (and for the optimal
value and the value at a candidate solution) of *data-based*, two-stage
stochastic programs. Unlike the other confidence-interval methods in
mpi-sppy, no distribution of the uncertain data is assumed: the estimators
work directly from sampled data. The methods and software are described in
[ChenWoodruff2023]_ and [ChenWoodruff2024]_.

.. note::

   This is the empirical (numpy-only) part of the package: the classical,
   extended, subsampling, and bagging methods. The *smoothed* methods, which
   depend on a distribution-fitting library, are merged separately; asking for
   a ``Smoothed_*`` method raises an informative error until then.

Modes
-----

There are two modes, each runnable with ``python -m``:

*User mode* (``user_boot``) computes a confidence interval for one problem
instance. A long list of arguments is supplied on the command line, so users
usually put the command in a shell script:

.. code-block:: bash

   $ python -m mpisppy.confidence_intervals.bootsp.user_boot module arguments

Here ``module`` is the name of an importable Python module (without ``.py``)
that supplies the scenario creator and helper functions, and ``arguments`` is
the list of double-dash options described below.

*Simulation mode* (``simulate_boot``) estimates the coverage rate of a method
over many replications; it is aimed at researchers. All options come from a
json file:

.. code-block:: bash

   $ python -m mpisppy.confidence_intervals.bootsp.simulate_boot instance.json

The model module
----------------

The named module must provide the usual mpi-sppy scenario-creation contract
plus a few helpers used by the bootstrap code:

* ``scenario_creator(scenario_name, ...)`` — build a Pyomo model for one
  (data) scenario, annotated as usual for mpi-sppy;
* ``scenario_names_creator(num_scens, start=None)`` — the list of scenario
  names;
* ``kw_creator(cfg)`` — keyword arguments for the scenario creator;
* ``inparser_adder(cfg)`` — add any model-specific options;
* ``xhat_generator(scenario_names, solver_name=None, ...)`` — solve for a
  candidate solution ``xhat`` when none is supplied. The bootstrap code looks
  for this fixed name first and falls back to the legacy
  ``xhat_generator_<module_name>``. If a precomputed ``xhat`` file is given
  (``--xhat-fname``) the generator is not called.

Methods
-------

The ``--boot-method`` (json ``boot_method``) option selects the estimator:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Token
     - Description
   * - ``Classical_gaussian``
     - Classical bootstrap, Gaussian confidence interval [eichhorn2007]_
   * - ``Classical_quantile``
     - Classical bootstrap, quantile confidence interval [eichhorn2007]_
   * - ``Extended``
     - Extended bootstrap [eichhorn2007]_
   * - ``Subsampling``
     - Subsampling bootstrap [eichhorn2007]_
   * - ``Bagging_with_replacement``
     - Bagging with replacement [lam2018]_
   * - ``Bagging_without_replacement``
     - Bagging without replacement [lam2018]_

Arguments
---------

Simulation and user modes use almost the same options; simulation mode reads
them from json (some with underscores), while user mode takes them on the
command line (with dashes). The main options are:

* ``max_count`` / ``--max-count`` — total sample size (integer).
* ``module_name`` — the module name (given as the first positional argument in
  user mode; a json key in simulation mode).
* ``candidate_sample_size`` / ``--candidate-sample-size`` — sample size used to
  compute ``xhat`` (M in the papers); ignored when an ``xhat`` file is given.
* ``sample_size`` / ``--sample-size`` — bootstrap/bagging sample size (N).
* ``subsample_size`` / ``--subsample-size`` — subsample size for bagging;
  ignored by the classical bootstrap methods.
* ``nB`` / ``--nB`` — number of bootstrap/bagging samples.
* ``alpha`` / ``--alpha`` — significance level (e.g. 0.05 for 95% confidence).
* ``seed_offset`` / ``--seed-offset`` — offset for the pseudo-random streams
  (enables replication); use 0 unless you have a reason not to.
* ``solver_name`` / ``--solver-name`` — solver (e.g. ``gurobi_direct``).
* ``xhat_fname`` / ``--xhat-fname`` — npy file with a precomputed ``xhat``, or
  the string ``"None"`` to compute it with ``xhat_generator``.
* ``optimal_fname`` (simulation only) — npy file with a (presumed) optimal
  value, or ``"None"`` to compute it from ``max_count`` scenarios.
* ``coverage_replications`` (simulation only) — number of coverage replications.
* ``boot_method`` / ``--boot-method`` — one of the tokens above.

There may also be model-specific options added by ``inparser_adder``.

Batch parallelism
-----------------

The bootstrap batches are split across MPI ranks and reassembled on rank 0
with ``Gatherv``, so a run can be accelerated with, e.g.,
``mpiexec -np 2 python -m mpi4py -m mpisppy.confidence_intervals.bootsp.user_boot ...``.
The estimate on rank 0 depends on the number of ranks because each rank seeds
its own bootstrap stream.

boot_general_prep
-----------------

``boot_general_prep`` writes the two npy files (a candidate ``xhat`` and a
presumed optimal value) that a simulation can reuse:

.. code-block:: bash

   $ python -m mpisppy.confidence_intervals.bootsp.boot_general_prep instance.json

Example
-------

The ``examples/bootsp/schultz`` directory has a small two-stage example whose
data is a deterministic function of the scenario number, so its results are
reproducible across solvers. From that directory:

.. code-block:: bash

   $ python -m mpisppy.confidence_intervals.bootsp.user_boot unique_schultz \
       --max-count 50 --candidate-sample-size 1 --sample-size 30 \
       --subsample-size 10 --nB 20 --alpha 0.1 --seed-offset 100 \
       --solver-name gurobi_direct --boot-method Classical_quantile

   $ python -m mpisppy.confidence_intervals.bootsp.simulate_boot unique_schultz.json

See ``examples/bootsp/schultz/schultz.bash`` for a serial run, a parallel run,
and a coverage simulation.

References
----------

.. [ChenWoodruff2023] Chen, X. and Woodruff, D.L.: Software for data-based
   stochastic programming using bootstrap estimation. INFORMS Journal on
   Computing (2023).

.. [ChenWoodruff2024] Chen, X. and Woodruff, D.L.: Distributions and Bootstrap
   for Data-based Stochastic Programming. Computational Management Science
   (2024).

.. [eichhorn2007] Eichhorn, A. and Romisch, W.: Stochastic integer
   programming: Limit theorems and confidence intervals. Mathematics of
   Operations Research, 32(1), 118-135 (2007).

.. [lam2018] Lam, H. and Qian, H.: Assessing solution quality in stochastic
   optimization via bootstrap aggregating. Winter Simulation Conference,
   2061-2071 (2018).
