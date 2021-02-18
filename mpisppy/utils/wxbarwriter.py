# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' An extension to save PH weights and/or PH xbar values to csv files.

    To use, specify either or both of the following keys to the PHoptions dict:

        "W_fname" : <str filename>
        "Xbar_fname": <str filename>

    If neither option is specified, this extension does nothing (prints a
    warning). If only one is specified, then only those values are saved.
    The resulting files will be output in the format:

        (W values) csv with rows: scenario_name, variable_name, weight_value
        (x values) csv with rows: variable_name, variable_value

    This format is consistent with the input required by WXbarReader.

    TODO:
        Check with bundles

    Written: DLW, July 2019
    Modified: DTM, Aug 2019
'''

import os
import pyomo.environ as pyo
import mpisppy.utils.wxbarutils
import mpisppy.extensions.extension

class WXBarWriter(mpisppy.extensions.extension.Extension):
    """ Extension class for writing the W values
    """
    def __init__(self, ph, rank, n_proc):
        # Check a bunch of files
        w_fname, x_fname, sep_files = None, None, False

        if ('W_fname' in ph.PHoptions): # W_fname is a path if separate_W_files=True
            w_fname = ph.PHoptions['W_fname']
        if ('Xbar_fname' in ph.PHoptions):
            x_fname = ph.PHoptions['Xbar_fname']
        if ('separate_W_files' in ph.PHoptions):
            sep_files = ph.PHoptions['separate_W_files']

        if (x_fname is None and w_fname is None and rank==0):
            print('Warning: no output files provided to WXBarWriter. '
                  'No values will be saved.')

        if (w_fname and (not sep_files) and os.path.exists(w_fname) and rank==0):
            print('Warning: specified W_fname ({fn})'.format(fn=w_fname) +
                  ' already exists. Results will be appended to this file.')
        elif (w_fname and sep_files and (not os.path.exists(w_fname)) and rank==0):
            print('Warning: path {path} does not exist. Creating...'.format(
                    path=w_fname))
            os.makedirs(w_fname, exist_ok=True)

        if (x_fname and os.path.exists(x_fname) and rank==0):
            print('Warning: specified Xbar_fname ({fn})'.format(fn=x_fname) +
                  ' already exists. Results will be appended to this file.')

        self.PHB = ph
        self.rank = rank
        self.w_fname = w_fname
        self.x_fname = x_fname
        self.sep_files = sep_files # Write separate files for each 
                                   # scenario's dual weights

    def pre_iter0(self):
        pass

    def post_iter0(self):
        pass
        
    def miditer(self, PHIter, conv):
        pass

    def enditer(self, PHIter):
        # Maybe you want to write here, but with an iteration specific name,
        # and maybe just for some iterations.
        pass

    def post_everything(self, PHIter, conv):
        if (self.w_fname):
            mpisppy.utils.wxbarutils.write_W_to_file(self.PHB, self.w_fname,
                sep_files=self.sep_files)
        if (self.x_fname):
            mpisppy.utils.wxbarutils.write_xbar_to_file(self.PHB, self.x_fname)

