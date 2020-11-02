.. _Extensions:

Extensions
==========

In order to allow for extension or modification of code behavior, many of
the components (hubs and spokes) support callout points. The objects
that provide the methods that are called are referred to as `extensions`.
Instead of using an extension, you could just hack in a function call,
but then every time ``mpi-sppy`` is updated, you would have to remember
to hack it back in. By using an extension object, the addition
or modification will remain available. Perhaps more important, the
extension can be included in some applications, but not others.

There are a number of extensions, particularly for PH, that are provided
with ``mpi-sspy`` and they provide examples that can be used for the
creation of more. Extensions can be found in ``mpisppy.extensions``.

PH extensions
-------------

Some of these can be used with other hubs. An extension object can be
passed to the PH constructor and it is assumed to have methods defined
for all the callout points in PH (so all of the examples do).  If you
want to use more than one extension, define a main extension that has
a reference to the other extensions and can call their methods in the
appropriate order. Extensions typicall access low level elements of
``mpi-sppy`` so writing your extensions is an advanced topic. We will
now describe a few of the extensions in the release.

mipgapper.py
^^^^^^^^^^^^

This is a good extension to look at as a first example. It takes a
dictionary with iteration numbers and mipgaps as input and changes the
mipgap at the corresponding iterations. The dictionary is provided in
the options dictionary in ``["gapperoptions"]["mipgapdict"]``.  There
is an example of its use in ``mpisppy.examples.sizes.sizes_demo.py``

fixer.py
^^^^^^^^

This extension provides methods for fixing variables (usually integers) for
which all scenarios have agreed for some number of iterations. There
is an example of its use in ``mpisppy.examples.sizes.sizes_demo.py``

xhat
^^^^

Most of the xhat methods can be used as an extension instead of being used
as a spoke, when that is desired (e.g. for serial applications).

WXBarWriter and WXBarReader
^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is an extension to write xbar and W values and another to read them.
An example of their use is shown in ``mpisppy.examples.sizes.sizes_demo.py``


