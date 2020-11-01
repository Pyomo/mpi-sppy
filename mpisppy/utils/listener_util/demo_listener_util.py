# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Demonstrate some uses of listener_util.py for asynchronous computing.
# This very silly and of limited value.
# DLW March 2019
# NOTE: If you have runtime error in this code, then you will need to kill
#       the listener threads.

import sys
import collections
import numpy as np
import mpi4py.MPI as mpi
from pyutilib.misc.timing import TicTocTimer
import time
import datetime as dt
import threading
import logging
import mpisppy.utils.listener_util.listener_util as listener_util

startdt = dt.datetime.now()

fullcomm = mpi.COMM_WORLD
rank = fullcomm.Get_rank()
n_proc = fullcomm.Get_size()

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )

def worker_bee(synchronizer, buzz=1e6, sting=100.0):
    """
    Do some busy work. Note that the synchronizer does not need to be an argument
    to this function (it could be accessed some other way).

    Args:
        sychronizer (object): used to do asynchronous sums
        buzz (int): Number of deviates to generate
        sting (float): standard deviation

    Returns:
        nothing (these worker functions usually want to update an object
                 but in this example we print).

    Note: To make this fun, we are randomly distribute the load.
          For even more fun, we occasionally try to 
          reduce the listner sleep time but the listener might not see it since
          it might sleep through the whole thing. This is subtle.
    
    We are going to track the time that each rank last reported just to show
    one way to do it. To spell out the way we will do it: 
    allocate n_proc of the vector to seconds_since_start and each rank 
    will put its seconds_since_start into its spot; hence, the sum will be 
    the report (since the others will have contributed zero).
    The number of times a rank has contributed to the sum could be tracked
    in an analogous way.
    """

    local_sum_sofar = 0
    local_iters_sofar = 0
    global_iters_sofar = 0
    old_sleep = np.zeros(1, dtype='d')
    old_sleep[0] = synchronizer.sleep_secs[0]

    # We will send/receive sum (1), iters (1) and secs (n_proc)
    # and we are going to concatenat all "vectors" into one.
    local_concat = {"FirstReduce": {"ROOT": np.zeros(2 + n_proc, dtype='d')}}
    global_concat = {"FirstReduce": {"ROOT": \
                      np.zeros(len(local_concat["FirstReduce"]["ROOT"]), \
                               dtype='d')}}

    # In this trivial example, we are going enable the side gig and
    # nothing will disable it. Normally, the listener side gig would
    # be expected to disable it.
    # Gratuitous call to enable the side_gig
    synchronizer.compute_global_data(local_concat,
                                     global_concat,
                                     rednames=["FirstReduce"],
                                     enable_side_gig = True)

    while global_iters_sofar < buzz:
        # attempt at less sleep, maybe (but don't do it twice in a row)
        if np.random.uniform() > 0.1 and old_sleep[0] \
           == synchronizer.sleep_secs[0]:
            old_sleep[0] = synchronizer.sleep_secs[0]
            synchronizer.sleep_secs[0] /= 10
            logging.debug ("TRYING to reduce sleep to {} from rank={}".\
                   format(synchronizer.sleep_secs[0], rank))
        elif old_sleep[0] != synchronizer.sleep_secs[0]:
            synchronizer.sleep_secs[0] = old_sleep[0]
            logging.debug ("putting sleep back to {} from rank={}".\
                   format(synchronizer.sleep_secs[0], rank))

        localiterstodo = int(np.random.uniform() * buzz / n_proc)
        if rank == 0:
            logging.debug("**rank 0: iterstodo="+str(localiterstodo))
        for i in range(localiterstodo):
            local_sum_sofar += np.random.normal(0, sting)
            
        local_iters_sofar += localiterstodo
        if rank == 0:
            logging.debug("rank 0: iterstodo {} iters sofar {} sum_so_far {}="\
                          .format(localiterstodo, local_iters_sofar, local_sum_sofar))
        local_concat["FirstReduce"]["ROOT"][0] = local_iters_sofar
        local_concat["FirstReduce"]["ROOT"][1] = local_sum_sofar
        local_concat["FirstReduce"]["ROOT"][2+rank] \
            = (dt.datetime.now() - startdt).total_seconds()

        # Only do "FirstReduce".
        synchronizer.compute_global_data(local_concat,
                                         global_concat,
                                         rednames=["FirstReduce"])

        global_iters_sofar = global_concat["FirstReduce"]["ROOT"][0]
        global_sum = global_concat["FirstReduce"]["ROOT"][1]
        if rank == 0:
            logging.debug("   rank 0: global_iters {} global_sum_so_far {}"\
                          .format(global_iters_sofar, global_sum))

    # tell the listener threads to shut down
    synchronizer.quitting = 1

    if rank == 0:
        print ("Rank 0 termination")
        print ("Based on {} iterations, the average was {}".\
               format(global_iters_sofar, global_sum / global_iters_sofar))
        print ("In case you are curious:\n rank \t last report *in* (sec)")
        for r in range(n_proc):
            print (r, "\t", global_concat["FirstReduce"]["ROOT"][2+r])
            

#=================================
def side_gig(synchro, msg = None):
    """ Demonstrate a listener side gig. This will be called by the listener.
        This is a small, silly function. Usually, the side-gig will check
        to see if it should do something or just pass.
        dlw babble: we can *look* at synchronizer.global_data for
        our redname because we know the reduction was just done.
        We can then jump in and modify local_data for the next reduction.
        This is either "hackish" or "c-like" depending on your point of view.
    Args:
        snchro (object): the Synchronizer object where the listener lives
        msg (str): just to show keyword arg; normally, this would be 
                   something (e.g., an object) that is modified
    """
    if synchro._rank == 0:
        print ("   (rank 0) ^*^* side_gig msg=", str(msg))
    logging.debug("enter side gig on rank %d" % rank)

    # Just to demonstrate how to do it, we will just return if nothing
    # changed in the first reduce.
    # NOTE: often the side will want to check and update enable_side_gig
    #       on the synchro object. See aph.
    # So this code needs to know the name of previous reduce.
    # BTW: in this example, we are treating the function as an object,
    #   but in most applications, this function will be in an object.
    prevredname = "FirstReduce"
    allthesame = True
    if len(side_gig.prev_red_prev_concat) == 0: # first time
        for cname, clen in synchro.Lens[prevredname].items():
            side_gig.prev_red_prev_concat[cname] = np.zeros(clen, dtype='d')
        allthesame = False # in case they are actually all zero

    for cname in synchro.Lens[prevredname]:
        if not np.array_equal(side_gig.prev_red_prev_concat[cname],
                              synchro.global_data[prevredname][cname]):
            allthesame = False
            break
    if allthesame:
        logging.debug("Skipping intermediate side_gig on rank %d" % rank)
        return

    logging.debug("Doing intermediate side_gig on rank %d" % rank)
    # It is OK for us to directly at global_data on the synchro because
    # the side_gig is "part of" the listener, the listener has the lock,
    # and only the listener updates global_data on the syncrho.
    # Side gigs are a bit of a hack.

    # this particular side_gig is very silly
    lastsideresult = synchro.global_data["SecondReduce"]["ROOT"]
    logging.debug("In case you are curious, before this listener call to"\
                  +"side_gig, the value of the secondreduce was {} on rank {}"\
                  .format(lastsideresult, rank))
    
    # dst, src
    np.copyto(side_gig.prev_red_prev_concat[cname],
              synchro.global_data[prevredname][cname])
    
    # For the second reduce, we are going to sum the ranks, which is silly.
    
    # Normally, the concats would be created once in an __init__, btw.
    local_concat = {"SecondReduce": {"ROOT": np.zeros(1, dtype='d')}}
    global_concat = {"SecondReduce": {"ROOT": np.zeros(1, dtype='d')}}
    local_concat["SecondReduce"]["ROOT"][0] = rank
    # We can (and should) do a dangerous put in the side_gig because
    # the listener will have the lock. If the worker gets to its compute_global
    # then it will have to wait for the lock.
    synchro._unsafe_put_local_data("SecondReduce", local_concat)


    # global data for the second reduce will be updated when we return to the
    # listener and
    # it will be available for the next get (we don't need a compute).
    # But it is not available now because the reduction has not been done yet.

side_gig.prev_red_prev_concat = {}#[cname]=synchro.global_data[prevredname][cname]
    

####### Main #######
usemsg = "usage: python demo_listener_util.py iters sleep seed; e.g.,\n" + \
          "python demo_listener_util.py 1e5 0.5 1134"
if len(sys.argv) != 4:
   raise RuntimeError(usemsg)
try:
    iters = int(sys.argv[1])
except:
   raise RuntimeError(sys.argv[1]+" is not a valid iters\n"+usemsg)    
try:
    sleep = float(sys.argv[2])
except:
   raise RuntimeError(sys.argv[2]+" is not a valid sleep\n"+usemsg)
try:
    seed = int(sys.argv[3])
except:
   raise RuntimeError(sys.argv[3]+" is not a valid seed\n"+usemsg)    

sting = 1 # standard devation
np.random.seed(seed)

# Note: at this point the first reduce is the only reduce
Lens = collections.OrderedDict({"FirstReduce": {"ROOT": 2+n_proc}})
if rank == 0:
    logging.debug("iters %d, sleep %f, seed %d".format((iters, sleep, seed)))
# "ROOT" is required to be the name of the global comm
synchronizer = listener_util.Synchronizer(comms = {"ROOT": fullcomm},
                                       Lens = Lens,
                                       work_fct = worker_bee,
                                       rank = rank,
                                       sleep_secs = sleep,
                                       asynch = True)
args = [synchronizer]
kwargs = {"buzz": iters, "sting": sting}
synchronizer.run(args, kwargs)

### now demo the use of a listener side gig between two reductions ###
if rank == 0:
    print ("testing side gig")
logging.debug("testing side gig on rank %d" % rank)

kwargs = {"msg": "Oh wow, the side gig is running"}
Lens = collections.OrderedDict({"FirstReduce": {"ROOT": 2+n_proc},
                                "SecondReduce": {"ROOT": 1}})
listener_gigs = {"FirstReduce": (side_gig, kwargs),
                 "SecondReduce": None}

synchronizer = listener_util.Synchronizer(comms = {"ROOT": fullcomm},
                                       Lens = Lens,
                                       work_fct = worker_bee,
                                       rank = rank,
                                       sleep_secs = sleep,
                                       asynch = True,
                                       listener_gigs = listener_gigs)
args = [synchronizer]
kwargs = {"buzz": iters, "sting": sting}
synchronizer.run(args, kwargs)
