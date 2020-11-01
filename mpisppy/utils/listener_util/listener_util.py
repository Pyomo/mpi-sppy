# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Async Utility. Started by Bill Hart,
#    simple generalization by Dave Woodruff 2019.
# See doc/philosophy.rst
# NOTE: DLW: as of February 2019, all we do is sum.......
# And if having the file named listener_util with a class named
# Synchronizer makes your head hurt, that's good. Get used to it.
"""
To avoid errors from Pyomo use,
                solve_keyword_args["use_signal_handling"] = False
"""
import numpy as np
import collections
import mpi4py.MPI as mpi
from pyutilib.misc.timing import TicTocTimer
import time
import cProfile
import threading
import logging

# yeah, globals are bad - being lazy in experimental code...
##timer = TicTocTimer()

class Synchronizer(object):
    """
    Manage both Async and Sync communcation.
    Args:
        comms (dict of mpi comms): keys are strings: mpi comms
               There must be a key "ROOT" that is the global comm
        Lens (sorted dict of dict of ints): 
                             the keys are reduction step names (defines them), 
                             then comms' keys. Contains length of vectors to sum
        work_fct (function): the main worker function (args can be supplied
                             when run is called.)
        rank (int): the mpi rank
        sleep_secs (float): Initial sleep seconds for the async listener.
        async (boolean): True for async, False for asynchronous; default False
        listener_gigs (dict of (fct, kwargs)): Optional. keys are step names,
                                                  functions after steps
                The functions are always passed this synchronizer object
                as the first arg, then whatever is in this dictionary's
                keyword args.
    Attributes:
        global_data (dict of np doubles): indexes match comms (advanced)
        sleep_secs (float): sleep seconds for the async listener
                            The listener takes the min from the reduce.
                            (not working as of March 2019)

        quitting (int): assign zero to cause the listeners to stop (async)

    Note:
        As of Python 3.7 async is a reserved word. Using asynch instead.
    """

    def __init__(self, comms, Lens, work_fct, rank,
                 sleep_secs, asynch=False, listener_gigs = None):
        self.asynch = asynch
        self.comms = comms
        self.Lens = Lens
        self._rank = rank
        self.sleep_secs = np.zeros(1, dtype='d')
        self.sleep_secs[0] = sleep_secs
        self.sleep_touse = np.zeros(1, dtype='d') # min over ranks
        # The side gigs are very dangerous.
        self.listener_gigs = listener_gigs
        self.enable_side_gig = False # TBD should be gigs & indexed by reduction
        
        self.global_quitting = 0
        self.quitting = 0
        self.work_fct = work_fct
        self.local_data = {}
        self.global_data = {}
        if not isinstance(self.Lens, collections.OrderedDict):
            raise RuntimeError("listener_util: Lens must be an OrderedDict")
        for redname in self.Lens.keys():
            self.local_data[redname] = {}
            self.global_data[redname] = {}
            for commname, ell in self.Lens[redname].items():
                assert(commname in self.comms)
                self.local_data[redname][commname] = np.zeros(ell, dtype='d')
                self.global_data[redname][commname] = np.zeros(ell, dtype='d')
        self.data_lock = threading.Lock()

    def run(self, args, kwargs):
        if self.asynch:
            print("ASYNC MODE - START")
            # THE WORKER
            wthread = threading.Thread(name=self.work_fct.__name__,
                                       target=self.work_fct,
                                       args = args,
                                       kwargs = kwargs)
            ## ph = threading.Thread(name='ph_main', target=ph_main)
            #ph.setDaemon(True)

            listenargs = [self._rank, self]
            l = threading.Thread(name='listener',
                                 target=Synchronizer.listener_daemon,
                                 args=listenargs)
            #l.setDaemon(True)

            l.start()
            wthread.start()

            l.join()
            print("ASYNC MODE - END")
        else:
            print("SYNC MODE - START")
            self.work_fct(*args, **kwargs)
            print("SYNC MODE - END")

    ###=====================###
    def _check_Lens(self, local_data_in, global_data_out, redname, cname):
        """ Essentially local to compute_global_data
        """
        if len(local_data_in[redname][cname]) \
           != self.Lens[redname][cname]:
            self.global_quitting = 1
            print ("\nERROR listener_util:cname={} len in={} Lens={}".\
                   format(cname, len(local_data_in[redname][cname]),
                          self.Lens[redname][cname]))
        if len(global_data_out[redname][cname]) \
           != self.Lens[redname][cname]:
            self.global_quitting = 1
            print ("\nERROR listener_util:cname={} len out={} Lens={}".\
                   format(cname, len(global_data_out[redname][cname]),
                          self.Lens[redname][cname]))

            
    #########################################################################
    def compute_global_data(self, local_data_in,
                            global_data_out,
                            enable_side_gig = False,
                            rednames=None,
                            keep_up=False):
        """
        Data processing. Cache local_data_in so it will be
        reduced later by the listener. Copy out the data that
        was the result of reductions the last time the listener reduced.

        NOTE:
            Do not call this from a listener side gig (lock issues).
        Args:
            local_data_in (dict): data computed locally
            global_data_out (dict): global version (often a sum over locals)
            enable_side_gig (boolean): sets a flag that allows the 
                   listener to run side gigs. It is intended to be a run once
                   authorization and the side gig code itself disables it.
            rednames (list of str): optional list of reductions to report
            keep_up (boolean):indicates the data out should include this data in;
                   otherwise, you will use a global that is "one notch behind."
        NOTE:
            as of April 2019. keep_up should be used  very sparingly.

        Note np.copy is dst, src
        """
        if self.asynch:
            logging.debug('Starting comp_glob update on Rank %d' % self._rank)
            self.data_lock.acquire() 
            logging.debug('Lock aquired by comp_glob on Rank %d' % self._rank)
            if rednames is None:
                reds = self.Lens.keys()
            else:
                reds = rednames
            for redname in reds:
                for cname in self.Lens[redname].keys():
                    if self.Lens[redname][cname] == 0:
                        continue
                    self._check_Lens(local_data_in, global_data_out,
                                     redname, cname)
                    if keep_up:
                        """ Capture the new data that will be summed.
                            Note: if you want this to be sparse, 
                            the caller should probably do it.
                        """
                        global_data_out[redname][cname] \
                            = self.global_data[redname][cname] \
                            - self.local_data[redname][cname] \
                            + local_data_in[redname][cname]
                        # The listener might sleep a long time, so update global
                        np.copyto(self.global_data[redname][cname],  
                                  global_data_out[redname][cname]) # dst, scr
                    else:
                        np.copyto(global_data_out[redname][cname],
                                  self.global_data[redname][cname]) 
                    # The next copy is done after global,
                    # so that keep_up can have the previous local_data.(dst, src)
                    np.copyto(self.local_data[redname][cname],
                              local_data_in[redname][cname]) 
            self.data_lock.release()
            logging.debug('Lock released on Rank %d' % self._rank)
            logging.debug('Ending update on Rank %d' % self._rank)
            if enable_side_gig:
                if self.enable_side_gig:
                    raise RuntimeError("side gig already enabled.")
                else:
                    self.enable_side_gig = True # the side_gig must disable
        else: # synchronous
            for redname in self.Lens.keys():
                for cname in self.Lens[redname].keys():
                    if self.Lens[redname][cname] == 0:
                        continue
                    comm = self.comms[cname]
                    comm.Allreduce([local_data_in[cname], mpi.DOUBLE],
                                   [global_data_out[cname], mpi.DOUBLE],
                                   op=mpi.SUM)
    ####################
    def get_global_data(self, global_data_out):
        """
        Copy and return the cached global data.

        Args:
            global_data_out (dict): global version (often a sum over locals)
        NOTE:
            As of March 2019, not used internally (duplicates code in compute)
        """
        if self.asynch:
            logging.debug('Enter get_global_data')
            logging.debug(' get_glob wants lock on Rank %d' % self._rank)
            self.data_lock.acquire() 
            logging.debug('Lock acquired by get_glob on Rank %d' % self._rank)
            for redname in self.Lens.keys():
                for cname in self.Lens[redname].keys():
                    if self.Lens[redname][cname] == 0:
                        continue
                    # dst, src
                    np.copyto(global_data_out[redname][cname],
                              self.global_data[redname][cname])
            self.data_lock.release()
            logging.debug('Lock released on Rank %d' % self._rank)
            logging.debug('Leave get_global_data')
        else:
            raise RuntimeError("get_global_data called for sycnhronous")

    ####################
    def _unsafe_get_global_data(self, redname, global_data_out):
        """
        NO LOCK. Copy and return the cached global data. Call only from
        within a side_gig!!

        Args:
            redname (string): the particular reduction name to copy
            global_data_out (dict): global version (often a sum over locals)
        NOTE:
            As of March 2019, not used internally (duplicates code in compute)
        """
        if self.asynch:
            logging.debug('Enter _usafe_get_global_data, redname={}'\
                          .format(redname))
            for cname in self.Lens[redname].keys():
                if self.Lens[redname][cname] == 0:
                    continue
                np.copyto(global_data_out[redname][cname],
                          self.global_data[redname][cname]) # dst, src
            logging.debug('Leave _unsafe_get_global_data')
        else:
            raise RuntimeError("_unsafe_get_global_data called for sycnhronous")

    ####################
    def _unsafe_put_local_data(self, redname, local_data_in):
        """
        NO LOCK. Copy and directly in to the local data cache. Call only from
        within a side_gig!!

        Args:
            redname (string): the particular reduction name to copy into
            local_data_in (dict): local data (often a summand over locals)
        NOTE:
            As of March 2019, not used internally (duplicates code in compute)
        """
        if self.asynch:
            logging.debug('Enter _usafe_put_local_data, redname={}'\
                          .format(redname))
            for cname in self.Lens[redname].keys():
                if self.Lens[redname][cname] == 0:
                    continue
                np.copyto(self.local_data[redname][cname],
                          local_data_in[redname][cname],) # dst, src
            logging.debug('Leave _unsafe_put_locobal_data')
        else:
            raise RuntimeError("_unsafe_put_local_data called for sycnhronous")


    @staticmethod
    def listener_daemon(rank, synchronizer):
        # both args added by DLW March 2019
        # listener side_gigs added by DLW March 2019.
        logging.debug('Starting Listener on Rank %d' % rank)
        while synchronizer.global_quitting == 0:
            # IDEA (Bill):  Add a Barrier here???
            synchronizer.data_lock.acquire()
            logging.debug('Locked; starting AllReduce on Rank %d' % rank)
            for redname in synchronizer.Lens.keys():
                for cname in synchronizer.Lens[redname].keys():
                    if synchronizer.Lens[redname][cname] == 0:
                        continue
                    comm = synchronizer.comms[cname]
                    logging.debug('  redname %s cname %s pre-reduce on rank %d' \
                                  % (redname, cname, rank))
                    comm.Allreduce([synchronizer.local_data[redname][cname],
                                    mpi.DOUBLE],
                                   [synchronizer.global_data[redname][cname],
                                    mpi.DOUBLE],
                                   op=mpi.SUM)
                    logging.debug(' post-reduce %s on rank %d' % (redname,rank))
                    if synchronizer.enable_side_gig \
                       and synchronizer.listener_gigs is not None \
                       and synchronizer.listener_gigs[redname] is not None:
                        args = [synchronizer]
                        fct, kwargs = synchronizer.listener_gigs[redname]
                        if kwargs is not None:
                            fct(*args, **kwargs)
                        else:
                            fct(*args)
            logging.debug('Still Locked; ending AllReduces on Rank %d' % rank)
            synchronizer.global_quitting = synchronizer.comms["ROOT"].allreduce(
                synchronizer.quitting, op=mpi.SUM)
            sleep_touse = np.zeros(1, dtype='d')
            sleep_touse[0] = synchronizer.sleep_secs[0]

            synchronizer.comms["ROOT"].Allreduce([synchronizer.sleep_secs,
                                                 mpi.DOUBLE],
                                                 [sleep_touse, mpi.DOUBLE],
                                                 op=mpi.MIN)

            logging.debug('  releasing lock on Rank %d' % rank)
            synchronizer.data_lock.release()
            logging.debug('  sleep for %f on Rank %d' % (sleep_touse, rank))
            try:
                time.sleep(sleep_touse[0])
            except:
                print("sleep_touse={}".format(sleep_touse))
                raise
        logging.debug('Exiting listener on Rank %d' % rank)


#synchronizer = Synchronizer(CropsLen, comm, sync=True)
#synchronizer.run()


