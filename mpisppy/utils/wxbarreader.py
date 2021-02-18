# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' An extension to initialize PH weights and/or PH xbar values from csv files.

    To use, specify either or both of the following keys to the PHoptions dict:

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

class WXBarReader(mpisppy.extensions.extension.Extension):
    """ Extension class for reading W values
    """
    def __init__(self, ph, rank, n_proc):

        ''' Do a bunch of checking if files exist '''
        w_fname, x_fname, sep_files = None, None, False
        if ('init_separate_W_files' in ph.PHoptions):
            sep_files = ph.PHoptions['init_separate_W_files']

        if ('init_W_fname' in ph.PHoptions):
            w_fname = ph.PHoptions['init_W_fname']
            if (not os.path.exists(w_fname)):
                if (rank == 0):
                    if (sep_files):
                        print('Cannot find path', w_fname)
                    else:
                        print('Cannot find file', w_fname)
                quit()

        if ('init_Xbar_fname' in ph.PHoptions):
            x_fname = ph.PHoptions['init_Xbar_fname']
            if (not os.path.exists(x_fname)):
                if (rank == 0):
                    print('Cannot find file', x_fname)
                quit()

        if (x_fname is None and w_fname is None and rank==0):
            print('Warning: no input files provided to WXBarReader. '
                  'W and Xbar will be initialized to their default values.')

        self.PHB = ph
        self.rank = rank
        self.w_fname = w_fname
        self.x_fname = x_fname
        self.sep_files = sep_files

    def pre_iter0(self):
        if (self.w_fname):
            mpisppy.utils.wxbarutils.set_W_from_file(
                    self.w_fname, self.PHB, self.rank,
                    sep_files=self.sep_files)
            self.PHB._reenable_W() # This makes a big difference.
        if (self.x_fname):
            mpisppy.utils.wxbarutils.set_xbar_from_file(self.x_fname, self.PHB)
            self.PHB._reenable_prox()

    def post_iter0(self):
        pass
        
    def miditer(self, PHIter, conv):
        ''' Called before the solveloop is called '''
        pass

    def enditer(self, PHIter):
        ''' Called after the solve loop '''
        pass

    def post_everything(self, PHIter, conv):
        pass
