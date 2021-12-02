.. _Sequential Sampling Confidence Intervals:

Sequential sampling
===================

Given a confidence interval, one can try to find a candidate solution
``xhat_one`` such that its optimality gap has this confidence interval.
The class ``SeqSampling`` in ``mpisppy.confiden_intervals.seqsampling.py`` implements three procedures described in 
[bm2011]_ and [bpl2012]_. It takes as an input a method to generate
candidate solutions and options, and its ``run`` method returns a ``xhat_one`` and a confidence interval on its optimality gap.

There are two stopping criterion supported with names based on the initials of
the authors who defined them: "BM" and "BPL".

Examples of use with the ``farmer`` problem and several options can be found in the main of ``seqsampling.py``. The following options dictionaries are illustrated:

- relative Width;

- fixed width, sequential;

- fixed width with stochastic samples.

  The keys used in the options dictionaries are taken directly from the corresponding paper, perhaps abbreviated in an obvious way. For example, the key `eps` corresponds to epsilon in the papers. 

For multi-stage, use `multi_seqsampling.py`.

Examples
--------

There is sample code for two-stage, sequential sampling in ``examples.farmer.farmer_seqsampling.py`` and
a bash scrip to test drive it is ``examples.farmer.farmer_sequential.bash``.

There is sample code for multi-stage, sequential sampling in ``examples.aircond.aircond_seqsampling.py`` and
a bash scrip to test drive it is ``examples.aircond.aircond_sequential.bash``.
