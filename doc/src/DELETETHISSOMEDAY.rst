Note for Documentation Writers
==============================

This rst file should be deleted by DLW some day.

* To make the documenation, use the command ``make html`` and then you can open ``_build\html\index.html`` to see it.

* rst files are sphinx files

* Code snippets: Don't include code snippets until we have doctests up and running. Until then, just reference example py files. (We should have doctests up in December at the latest)

* I would like to keep the version on master ready to ship to readthedocs at a moments notice, so don't leave TODOs or TBDs or XXXX in PRs. Once we establish the connection, BTW, readthedocs will automatically pull from master, so master should always "look good."
