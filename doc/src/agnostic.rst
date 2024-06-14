AML Agnosticism
===============

The mpi-sppy package provides callouts so that algebraic modeling languages
(AMLs) other than Pyomo can be used. A growing number of AMLs are supported
as `guest` languages (we refer to mpi-sppy as the `host`).

From the end-user's perspective
-------------------------------

When mpi-sppy is used for a model developed in an AML for which support
has been added, the end-user runs the ``mpisppy.agnostic.agnostic_cylinders.py``
program which serves as a driver that takes command line arguments and
launches the requested cylinders.  The file
``mpisppy.agnostic.go.bash`` provides examples of a few command lines.


From the modeler's perspective
------------------------------

Assuming support has been added for the desired AML, the modeler supplies
two files:

- a model file with the model written in the guest AML (AMPL example: ``mpisppy.agnostic.examples.farmer.mod``)
- a thin model wrapper for the model file written in Python (AMPL example: ``mpisppy.agnostic.examples.farmer_ampl_model.py``). This thin python wrapper is model specific.

There can be a little confusion if there are error messages because
both files are sometimes refered to as the `model file.`

Most modelers will probably want to import the deterministic guest model into their
python wrapper for the model and the scenario_creator function in the wrapper
modifies the stochastic paramaters to have values that depend on the scenario
name argument to the scenario_creator function.

(An exception is when the guest is in Pyomo, then the wrapper
file might as well contain the model specification as well so
there typically is only one file.)


From the developers perspective
-------------------------------

If support has not yet been added for an AML, it is almost easier to
add support than to write a guest interface for a particular model. To
add support for a language, you need to write a general guest
interface in Python for it (see, e.g., ampl_guest.py or
pyomo_guest.py) and you need to add/edit a few lines in
``mpisppy.agnostic.agnostic_cylinders.py`` to allow end-users to
access it.


Special Note for developers
---------------------------

The general-purpose guest interfaces might not be the fastest possible
for many guest languages because they don't use indexes from the
original model when updating the objective function. If this is an issue,
you might want to write a problem-specific module to replace the guest
interface and the model wrapper with a single module. For an example, see
``examples.farmer.farmer_xxxx_agnostic``, where xxxx is replaced,
e.g., by ampl. 
