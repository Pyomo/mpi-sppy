.. _Sequential Sampling Confidence Intervals:

Sequential sampling
===================

Similarly, given an confidence interval, one can try to find a candidate solution
``xhat_one`` such that its optimality gap has this confidence interval.
The class ``SeqSampling`` implements three procedures described in 
[bm2011]_ and [bpl2012]_. It takes as an input a method to generate
candidate solutions and options, and returns a ``xhat_one`` and a confidence interval on its optimality gap.

There are two stopping criterion supported with names based on the initials of
the authors: "BM" and "BPL".

Examples of use with the ``farmer`` problem and several options can be found in the main of ``seqsampling.py``. The following options dictionaries are illustrated:

- Relative Width

- fixed width, sequential

- fixed width with stochastic samples.
