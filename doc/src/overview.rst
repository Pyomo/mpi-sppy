.. _Overview:

Overview
=========


Roles
-----

We take that view that there are the following roles (that might be
filled by one person, or multiple people):

- *End-User*: runs a program to get the results of optimization under uncertainty.
- *Modeler*: creates the Pyomo model and establishes the nature of the scenario tree.
- *Developer*: writes the program that the end-user runs and uses the model(s) created by the modeler.
- *Contributors to mpi-sppy*: the creators of, and contributors to, the mpi-sppy library.

If you are reading this document, we assume that you are doing so in your
role as a developer, or perhaps in your role as a modeler. Neither this
document, nor ``mpi-sppy`` are written with the intention that they will
be employed directly by end-users.

Basics
------

The ``mpi-sppy`` library is based on the idea that one starts with
deterministic ``Pyomo`` model and extends it to accommodate uncertainty.


Cylinders
---------

To achieve results with the lowest possible wall-clock time,
``mpi-sppy`` make use of a hub-and-spoke architecture. The hub controls the
progression of the optimization and executes the "main" optimization algorithm.
Each spoke performs one (broad) task, such as computing a bound, or
producing heuristic solutions. We refer to the hub or a spoke generically as a
`cylinder`. A cylinder is composed of multiple
`ranks` (which is the name given by
MPI to compute units).  The ranks communicate asynchronously between
the cylinders and in whatever manner is appropriate within a
cylinder. We often make use of MPI *reductions* within a cylinder
that require at worst :math:`n \log(n)` effort, where `n` denotes the
number of ranks in the cylinder.

While the architecture is oriented towards scenario-based uncertainty,
it supports implementation of both scenario-based and stage-based
decomposition algorithms for stochastic programming. In addition to
core decomposition patterns, the architecture enables
exploitation of large numbers of parallel compute units for
speculative computations, e.g., distinct and diverse strategies for
computing upper and lower bounds.

Most developers using ``mpi-sppy`` will not need to concern themselves
very much with the architecure because ``mpi-sppy`` can take
care of the communcation aspects.

Solver threads
--------------

To illustrate one of the issues associated with having a lot of things
running at once, we note that you will often want to limit your
solvers to two threads if they take essentially as many threads as
they can get (e.g., cplex and gurobi). This is because if you have
multiple cylinders each launching a solver and each solver wants to
use every thread it can find, you may oversubscribe you computer
rather badly.
