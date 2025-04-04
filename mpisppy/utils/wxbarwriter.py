###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
''' An extension to save PH weights and/or PH xbar values to csv files.

    To use, specify either or both of the following keys to the options dict:

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
import mpisppy.utils.wxbarutils
import mpisppy.extensions.extension

import mpisppy.MPI as MPI 
                         
n_proc = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

def add_options_to_config(cfg):
    cfg.add_to_config("W_and_xbar_writer",
                      description="Enables the w and xbar writer(default False)",
                      domain=bool,
                      default=False)
    cfg.add_to_config("W_fname",
                      description="Path of final W file (default None)",
                      domain=str,
                      default=None)
    cfg.add_to_config("Xbar_fname",
                      description="Path of final Xbar file (default None)",
                      domain=str,
                      default=None)
    cfg.add_to_config("separate_W_files",
                      description="If True, writes W to separate files (default False)",
                      domain=bool,
                      default=False)            

class WXBarWriter(mpisppy.extensions.extension.Extension):
    """ Extension class for writing the W values
    """
    def __init__(self, ph):

        assert 'cfg' in ph.options
        self.cfg = ph.options['cfg']
        if self.cfg.get("W_and_xbar_writer") is None or not self.cfg.W_and_xbar_writer:
            self.active = False
            return  # nothing to do here
        else:
            self.active = True
        
        # Check a bunch of files
        w_fname, x_fname, sep_files = self.cfg.W_fname, self.cfg.Xbar_fname, self.cfg.separate_W_files

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
        self.cylinder_rank = rank
        self.w_fname = w_fname
        self.x_fname = x_fname
        self.sep_files = sep_files # Write separate files for each 
                                   # scenario's dual weights

    def pre_iter0(self):
        pass

    def post_iter0(self):
        pass
        
    def miditer(self):
        pass

    def enditer(self):
        if not self.active:
            return  # nothing to do.
        else:
            pass
        #if (self.w_fname):
        #     fname = f'fname{self.PHB._PHIter}.csv'
        #     mpisppy.utils.wxbarutils.write_W_to_file(self.PHB, w_fname,
        #         sep_files=self.sep_files)


    def post_everything(self):
        if not self.active:
            return  # nothing to do.
        if (self.w_fname):
            fname = self.w_fname
            mpisppy.utils.wxbarutils.write_W_to_file(self.PHB, fname,
                sep_files=self.sep_files)
        if (self.x_fname):
            mpisppy.utils.wxbarutils.write_xbar_to_file(self.PHB, self.x_fname)

