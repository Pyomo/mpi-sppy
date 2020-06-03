Overview of use
===============

Communication via c-like vectors.

Create a synchronizer object. In this document, we will call it
`synchronizer`.

shutdown
^^^^^^^^

Set `synchronizer.quitting` to one to cause listener threads to shutdown.

If there is an error or an assert failure, etc. you will may need to kill
some threads.
