.. _Sequential Sampling Confidence Intervals:

Sequential Sampling (MRP)
=========================

Sequential sampling (the Multiple Replication Procedure, or MRP)
finds a candidate solution ``xhat`` and a confidence interval on
its optimality gap by solving a sequence of approximate problems
with increasing sample sizes.  Two stopping criteria are
supported, named after the authors who defined them:

- **BM** (Bayraksan and Morton [bm2011]_): relative width criterion
- **BPL** (Bayraksan and Pierre-Louis [bpl2012]_): fixed width criterion

Using ``mrp_generic.py``
-------------------------

The recommended way to run sequential sampling is with
``mpisppy.mrp_generic``.  Like ``generic_cylinders.py``, it takes
a ``--module-name`` argument pointing to your model module.

Your model module must provide the same functions required by
``generic_cylinders``: ``scenario_creator``, ``scenario_names_creator``,
``kw_creator``, ``inparser_adder``, and ``scenario_denouement``.
See :ref:`helper_functions` for details.

Two-stage example (BM criterion)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -m mpisppy.mrp_generic --module-name farmer \
        --num-scens 3 \
        --solver-name gurobi \
        --stopping-criterion BM \
        --BM-h 2.0 \
        --BM-hprime 0.5 \
        --BM-eps 0.5 \
        --BM-eps-prime 0.4 \
        --BM-p 0.2 \
        --BM-q 1.3 \
        --confidence-level 0.95

Two-stage example (BPL criterion)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -m mpisppy.mrp_generic --module-name farmer \
        --num-scens 3 \
        --solver-name gurobi \
        --stopping-criterion BPL \
        --BPL-eps 100.0 \
        --BPL-c0 25

Multi-stage example
^^^^^^^^^^^^^^^^^^^^

For multi-stage problems, supply ``--branching-factors``:

.. code-block:: bash

    python -m mpisppy.mrp_generic --module-name aircond \
        --branching-factors "3 3 2" \
        --solver-name gurobi \
        --stopping-criterion BM \
        --BM-h 0.55 --BM-hprime 0.5 \
        --BM-eps 0.5 --BM-eps-prime 0.4 \
        --BM-p 0.2 --BM-q 1.2

Choosing how xhat is generated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, each candidate solution is computed by solving the
extensive form (EF) for the current sample.  For problems too
large for EF, use hub-and-spoke decomposition instead:

.. code-block:: bash

    mpiexec -np 3 python -m mpi4py mpisppy/mrp_generic.py \
        --module-name farmer \
        --num-scens 3 \
        --solver-name gurobi \
        --xhat-method cylinders \
        --stopping-criterion BM \
        --default-rho 1 --max-iterations 10 \
        --lagrangian --xhatshuffle \
        --BM-h 2.0 --BM-hprime 0.5 \
        --BM-eps 0.5 --BM-eps-prime 0.4 \
        --BM-p 0.2 --BM-q 1.3

When ``--xhat-method cylinders`` is selected, the decomposition
arguments (PH options, spoke selection, rho settings, etc.) are
the same as for ``generic_cylinders``.

.. note::
   ``--xhat-method EF`` does not require MPI.
   ``--xhat-method cylinders`` requires ``mpiexec``.

Command-line reference
-----------------------

**Required:**

- ``--module-name``: model module (same as ``generic_cylinders``)
- ``--solver-name``: solver to use
- ``--stopping-criterion``: ``BM`` or ``BPL``

**Sequential sampling options:**

- ``--confidence-level``: 1 - alpha (default 0.95)
- ``--sample-size-ratio``: ratio of xhat sample size to gap estimator
  sample size (default 1.0)
- ``--ArRP``: number of estimators to pool (default 1)
- ``--mrp-max-iterations``: safety cap on iterations (default 200)
- ``--xhat-method``: ``EF`` (default) or ``cylinders``

**BM stopping criterion options** (see [bm2011]_):

- ``--BM-h``: controls width of confidence interval (default 1.75)
- ``--BM-hprime``: tradeoff between width and sample size (default 0.5)
- ``--BM-eps``: controls termination (default 0.2)
- ``--BM-eps-prime``: controls termination (default 0.1)
- ``--BM-p``: controls sample size growth (default 0.1)
- ``--BM-q``: related to sample size growth (default 1.2)

**BPL stopping criterion options** (see [bpl2012]_):

- ``--BPL-eps``: controls termination (default 1)
- ``--BPL-c0``: starting sample size (default 20)
- ``--BPL-n0min``: if nonzero, enables stochastic sampling (default 0)

**Output:**

- ``--solution-base-name``: write xhat to ``<name>.npy``

Output
------

``mrp_generic`` prints the number of iterations, the confidence
interval on the optimality gap, and (optionally) writes the
candidate solution to a ``.npy`` file.  The result is a dictionary
with keys:

- ``T``: number of sequential sampling iterations
- ``Candidate_solution``: the xhat dict (e.g. ``{'ROOT': [...]}``)
- ``CI``: confidence interval ``[0, upper_bound]`` on the gap


For Developers
--------------

The classes and modules described below are what ``mrp_generic``
uses internally.  Most users should use ``mrp_generic`` directly
rather than these lower-level interfaces.

``SeqSampling`` class
^^^^^^^^^^^^^^^^^^^^^^

The class ``SeqSampling`` in
``mpisppy.confidence_intervals.seqsampling.py`` implements the BM
and BPL procedures from [bm2011]_ and [bpl2012]_.  It takes as
input a reference model name, an ``xhat_generator`` callback, a
``Config`` object with stopping criterion parameters, and returns a
candidate solution with a confidence interval on its optimality gap.

The ``xhat_generator`` must have the signature::

    def xhat_generator(scenario_names, solver_name=None,
                       solver_options=None, **kwargs) -> dict

It receives scenario names and solver info, solves the approximate
problem, and returns a nonant cache (a dict mapping node names to
lists of values, e.g. ``{'ROOT': [v1, v2, ...]}``).

``mrp_generic`` provides two generic generators:
``_ef_xhat_generator`` (uses ``Amalgamator``) and
``_cylinder_xhat_generator`` (uses ``WheelSpinner``).

``IndepScens_SeqSampling`` class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For multi-stage problems, ``IndepScens_SeqSampling`` in
``mpisppy.confidence_intervals.multi_seqsampling.py`` extends
``SeqSampling`` to use independent scenarios instead of a single
scenario tree.  ``mrp_generic`` selects this class automatically
when ``--branching-factors`` is supplied.

Options dictionaries
^^^^^^^^^^^^^^^^^^^^^

The keys used in the options are taken directly from the
corresponding papers, perhaps abbreviated in an obvious way.  For
example, ``BM_eps`` corresponds to epsilon in [bm2011]_.

The ``Config`` object passed to ``SeqSampling`` can be populated
using the helper functions in
``mpisppy.confidence_intervals.confidence_config``:

- ``confidence_config(cfg)`` — adds ``confidence_level``
- ``sequential_config(cfg)`` — adds ``sample_size_ratio``, ``ArRP``,
  ``kf_GS``, ``kf_xhat``
- ``BM_config(cfg)`` — adds BM parameters
- ``BPL_config(cfg)`` — adds BPL parameters

Model-specific examples
^^^^^^^^^^^^^^^^^^^^^^^^

There are model-specific sequential sampling examples that
predate ``mrp_generic`` (these build their own
``xhat_generator`` and wire up the ``SeqSampling`` class
directly):

- Two-stage: ``examples/farmer/CI/farmer_seqsampling.py``
  (bash driver: ``examples/farmer/CI/farmer_sequential.bash``)
- Multi-stage: ``examples/aircond/aircond_seqsampling.py``
  (bash driver: ``examples/aircond/aircond_sequential.bash``)
