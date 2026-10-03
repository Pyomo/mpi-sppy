Notes for Code Contributors
===========================

Tests
^^^^^

There are some tests run by github, but before submitting a PR, you should
cd to the examples directory to look at and run ``run_all.py``. Notice that
``run_all.py`` creates some timing benchmark csv files as a side-effect.
As a general rule, you should leave them in your local examples directory
for regression testing and do not push them to the main repository.

Extensions, convergers and checkpointing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A new extension or converger must say what happens to its state when a run is
checkpointed and resumed: implement ``checkpoint_state()`` and
``restore_state(state)``, or set ``checkpoint_stateless = True`` on the class.
:ref:`checkpointing_your_extension` shows both. ``TestShippedExtensionsAnswerTheQuestion`` in
``mpisppy/tests/test_checkpoint_extensions.py`` fails for an extension or
converger in the repository that does neither.

If yours keeps state, add a case to ``test_checkpoint_extensions.py`` that
subclasses ``_ABMixin``. Give it ``ext_classes()``; its ``run_ab()`` runs
your extension straight through and again with a stop and a resume, and
``assert_bit_identical`` compares the two. Check the restored state itself as
well, so the test cannot pass just because the model reaches the same iterate
without it. ``TestSlammerResume`` is a short example.

import mpi
^^^^^^^^^^

Do not import mpi4py directly. Use

::

   from mpisppy import MPI

You will have access to `MPI.COMM_WORLD` as a result.
