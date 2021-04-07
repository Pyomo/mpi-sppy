Design Philosophy
=================

This is an mpi package, therefor it uses ranks and comms.

.. warning:
   This is under rapid development as of March 2019.

.. note:
   This does not support Python 2.7 and might not support anything below 3.5.

Layers
^^^^^^

The listener should mostly sleep. When it wakes up it grabs the lock
and goes through a series of reductions with small amounts of work in
between.  The reductions may take place over multiple comms.

Of course, there could be only only one reduction per listener wakeup,
there could be only one comm, and the listener does not have to do any
work. This is the case for PH on a two-stage problem (for multi-stage
problems PH uses multiple comms.

The reason we do a series of reductions is that for applications like
APH, it is needed so we don't have to wait for a long sleep to finish
what should be done at "essentially" a single barrier.

If all you want is one reduction (e.g. for PH), then all the redname
stuff is just a distraction.


Exceptions
^^^^^^^^^^

If you get an exception, you may not see it! Furthermore, things may
need to be killed.

xxx what's up with the sleep time? 0.5 seems to be way faster than 1/2 a
second
