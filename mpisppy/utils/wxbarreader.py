###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
''' An extension to initialize PH weights and/or PH xbar values from csv files.

    To use, specify either or both of the following keys to the options dict:

        "init_W_fname" : <str filename>
        "init_Xbar_fname": <str filename>

    If neither option is specified, this extension does nothing (i.e. is
    silent--does not raise a warning/error). If only one is specified, then
    only those values are initialized. The specified files should be
    formatted as follows:

        (W values) csv with rows: scenario_name, variable_name, weight_value
        (x values) csv with rows: variable_name, variable_value

    Rows that begin with "#" are treated as comments. If the files are missing
    any values, raises an error. Extra values are ignored.

    TODO:
        Check with bundles.

    Written: DLW, July 2019
    Modified: DTM, Aug 2019
'''

import mpisppy.utils.wxbarutils
import os # For checking if files exist
import mpisppy.extensions.extension
import mpisppy.MPI as MPI

n_proc = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

def add_options_to_config(cfg):

    cfg.add_to_config("W_and_xbar_reader",
                      description="Enables the w and xbar reader (default False)",
                      domain=bool,
                      default=False)
    cfg.add_to_config("init_W_fname",
                      description="Path of initial W file (default None)",
                      domain=str,
                      default=None)
    cfg.add_to_config("init_Xbar_fname",
                      description="Path of initial Xbar file (default None)",
                      domain=str,
                      default=None)
    cfg.add_to_config("init_separate_W_files",
                      description="If True, W is read from separate files (default False)",
                      domain=bool,
                      default=False)

class WXBarReader(mpisppy.extensions.extension.Extension):
    """ Extension class for reading W values
    """
    def __init__(self, ph):

        assert 'cfg' in ph.options
        self.cfg = ph.options['cfg']
        if self.cfg.get("W_and_xbar_reader") is None or not self.cfg.W_and_xbar_reader:
            self.not_active = True
            return  # nothing to do here
        else:
            self.not_active = False

        ''' Do a bunch of checking if files exist '''
        w_fname, x_fname, sep_files = self.cfg.init_W_fname, self.cfg.init_Xbar_fname, self.cfg.init_separate_W_files

        if w_fname is not None:
            if (not os.path.exists(w_fname)):
                if (rank == 0):
                    if (sep_files):
                        print('Cannot find path', w_fname)
                    else:
                        print('Cannot find file', w_fname)
                quit()

        if x_fname is not None:
            if (not os.path.exists(x_fname)):
                if (rank == 0):
                    print('Cannot find file', x_fname)
                quit()

        if (x_fname is None and w_fname is None and rank==0):
            print('Warning: no input files provided to WXBarReader. '
                  'W and Xbar will be initialized to their default values.')

        self.PHB = ph
        self.cylinder_rank = rank
        self.w_fname = w_fname
        self.x_fname = x_fname
        self.sep_files = sep_files

    def pre_iter0(self):
        pass


    def post_iter0(self):
        pass

    def miditer(self):
        ''' Called before the solveloop is called '''
        if self.not_active:
            return  # nothing to do.
        if self.PHB._PHIter == 1:
            if self.w_fname:
                mpisppy.utils.wxbarutils.set_W_from_file(
                        self.w_fname, self.PHB, self.cylinder_rank,
                        sep_files=self.sep_files)
                self.PHB._reenable_W() # This makes a big difference.
            if self.x_fname:
                mpisppy.utils.wxbarutils.set_xbar_from_file(self.x_fname, self.PHB)
                self.PHB._reenable_prox()

    def enditer(self):
        ''' Called after the solve loop '''
        pass

    def post_everything(self):
        pass
