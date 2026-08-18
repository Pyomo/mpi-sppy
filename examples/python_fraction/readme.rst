Fraction of time in python
==========================

See ``python_fraction.tex`` for the writeup and the results.

Files
-----

``run_experiments.bash``
    Runs every case under scalene, repeating each case so that the writeup can
    show run-to-run spread instead of a single sample. Repetitions matter here:
    scalene works by sampling, so one run of one case is weak evidence.

``scalene_wrapper.bash``
    Rank-aware scalene launcher; ``mpiexec`` runs this rather than python
    directly, so that each rank can name its own output file. Not normally run
    by hand.

``summarize_reps.py``
    Aggregates the repetitions into the LaTeX tables.

``make_tables.bash``
    Regenerates the tables from profiles already on disk. No experiments rerun.

``make_scalene_latex_table.py``
    Detail table for a single run (one directory of per-rank profiles).

``scalene_totals.py``
    Pulls the Python/native/system split out of a scalene JSON profile. Shared
    by the two table generators.

Running
-------

Profile the default case list with the persistent solver interface, three
repetitions each::

    $ SOLVER=gurobi_persistent ./run_experiments.bash

Then, to get the unprofiled wall times that the overhead column needs::

    $ SOLVER=gurobi_persistent PROFILE=0 ./run_experiments.bash

And regenerate the tables::

    $ ./make_tables.bash

Individual cases can be named on the command line, e.g.
``./run_experiments.bash farmer60 sslp_5_25_50``.

Notes
-----

Prefer a persistent solver interface. ``--solver-name gurobi`` is Pyomo's
file-based interface, which writes an LP file and parses a solution file in
Python on every solve; on small subproblems that Python work dominates
everything else, and the measured Python fraction then says more about Pyomo's
file writer than about mpi-sppy.

The numbers come from the scalene JSON rather than from scraping
``scalene view --cli``. Reading the JSON avoids three problems with scraping:
the CLI rounds each line to a whole percent, ``--reduced`` hides low-usage
lines so the sums undercount, and the output is colourized, which silently
broke the original row-matching regex. ``make_scalene_latex_table.py --from-cli``
still parses the CLI if you want to compare the two paths.

Scalene occasionally dies during startup with a ``KeyError`` from inside
``importlib``, before any mpi-sppy code runs; ``run_experiments.bash`` retries a
repetition that comes back with fewer profiles than ranks.

This code suite is probably fragile because scalene seems to do major updates
that change the output format. It was last run against scalene 2.0.1.
