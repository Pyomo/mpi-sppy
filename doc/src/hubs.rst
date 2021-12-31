.. _Hubs:

Hubs
====

In this section we describe some of the hub classes that are part of
the ``mpi-sppy`` release.  Many of these hubs have hooks for extension
and modification provided in the form of :ref:`Extensions`.  Many of
the algorithms can be run in stand-alone mode (not as a hub with
spokes), which is briefly described in :ref:`Drivers`.  Most hubs have
an internal convergence metric, but the threshold option
(``--intra-hub-conv-thresh`` in ``baseparsers.py`` or ``"convthresh"``
in ``PHoptions``) is often set to a negative number so internal
convergence is ignored in favor of the threshhold on the gap between
upper and lower bounds as computed by the spokes (``--rel-gap`` and
``--abs-gap`` in ``baseparsers.py``).  Most hubs can be terminated
based on an iteration limit (``--max-iterations`` in ``baseparsers.py``).

PH
--

The PH implementation can be used with most spokes because it can
supply x and/or W values at every iteration and numerous extensions
are in the release.  It supports a full set of extension callout points.

.. _linearize_proximal:

Linearize proximal terms
^^^^^^^^^^^^^^^^^^^^^^^^

The proximal term can be approximated linearly using the PHoption
`linearize_proximal_terms` (which is included as
``--linearize-proximal-terms`` in ``baseparsers.py``). If this option
is specified, then the option `proximal_linearization_tolerance`
(which is ``--proximal-linearization-tolerance`` in
``baseparsers.py``) is a parameter.  A cut will be added if the
proximal term approximation is looser than this value (default 1e-1).


If only the binary terms should be 
approximated, the option `linearize_binary_proximal_terms` can be used. 

lshaped
-------

The L-shaped algorithm decomposes by stage and the current implementation is
for two-stage problems only.

cross_scen_hub
--------------

The cross scenario hub supports only two-stage problems at this time.

APH
---

The implementation of Asynchronous Projective Hedging is described in a
forthcoming paper.

Hub Convergers
--------------

Execution of mpi-sppy programs can be terminated based on a variety of criteria.
The simplest is hub iteration count and the most common is probably relative
gap between upper and lower bounds. It is also possible to terminate
based on convergence within the hub; furthermore, convergence metrics within
the hub can be helpful for tuning algorithms.

The scenario decomposition methods (PH and APH) allow for optional
metrics to be used as plug-ins. A pattern that can be followed is shown
in the farmer example. The ``farmer_cylinders.py`` file has::

   from mpisppy.convergers.norm_rho_converger import NormRhoConverger

and optionally passes ``NormRhoConverger`` to the hub constructor. Note that you can observe
the behavior of the hub converger using the option ``--with-display-convergence-detail``.

Unfortunately, the word "converger" is also used to describe spokes that return bounds
for the purpose of measuring overall convergence (as opposed to convergence within the hub
algorithm.)  This word is used fairly deep in the code to distinguish spokes
that return bounds.

