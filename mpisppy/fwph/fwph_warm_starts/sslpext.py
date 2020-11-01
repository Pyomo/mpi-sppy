# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
from mpisppy.utils.wxbarreader import WXBarReader
from mpisppy.utils.wxbarwriter import WXBarWriter
import numpy as np
import pyomo.environ as pyo

import mpisppy.extension

class SSLPExtension(mpisppy.extension.Extension):

    def __init__(self, ph, rank, n_proc):
        self.ph = ph
        self.rank = rank
        self.reader_object = WXBarReader(ph, rank, n_proc)
        self.writer_object = WXBarWriter(ph, rank, n_proc)

    def pre_iter0(self):
        self.reader_object.pre_iter0()
        self.writer_object.pre_iter0()
                                        
    def post_iter0(self):
        self.reader_object.post_iter0()
        self.writer_object.post_iter0()
        print_soln_codes(self.ph._PHIter)

    def miditer(self, PHIter, conv):
        self.reader_object.miditer(PHIter, conv)
        self.writer_object.miditer(PHIter, conv)
        if (self.rank == 0 and PHIter == 1):
            print(f'Trival bound: {self.ph.trivial_bound:.4f} '
                   '(Only valid if no prox term is used in iter0)') 

    def enditer(self, PHIter):
        self.reader_object.enditer(PHIter)
        self.writer_object.enditer(PHIter)
        print_soln_codes(PHIter)

    def post_everything(self, PHIter, conv):
        self.reader_object.post_everything(PHIter, conv)
        self.writer_object.post_everything(PHIter, conv)

    def print_soln_codes(PHIter):
        codes = []
        for (name, scenario) in self.ph.local_scenarios.items():
            code = get_soln_code(scenario)
            codes.append(f'{code:8d}')
        # print(f'{PHIter:3d}', ' '.join(codes))
        if (self.rank == 0):
            print(PHIter)

    def get_soln_code(scenario):
        root = scenario._PySPnode_list[0] # Only one node b/c two-stage
        var  = np.array([pyo.value(v) for v in root.nonant_vardata_list])
        strr = ''.join(['1' if np.abs(1-val) < 1e-7 else '0' for val in var])
        code = int(strr, 2)
        return code

if __name__=='__main__':
    print('No main for sslpext.py')
