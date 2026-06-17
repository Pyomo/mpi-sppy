The battery example was never fully developed. Nothing in the
documentation, `run_all.py`, `afew.py`, `generic_tester.py`, or the
test suite exercised it, and `batteryext.py` imported a module
(`mpisppy.extension`) that no longer exists at HEAD, so it also did
not import cleanly.

The original sources have been moved to ``archive/`` for history
rather than deleted. If you want to revive this example, the natural
path is to add the hooks expected by
``mpisppy/generic_cylinders.py`` (``scenario_names_creator``,
``kw_creator``, ``inparser_adder``) to ``battery.py`` and wire
whatever is needed from ``batteryext.py`` against the current
``mpisppy.extensions.extension.Extension`` base class.
