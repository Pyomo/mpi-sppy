# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import mpisppy.extension

class BatteryExtension(mpisppy.extension.Extension):

    def __init__(self, ph, rank, n_proc):
        self.rank = rank
        self.ph = ph

    def pre_iter0(self):
        pass

    def post_iter0(self):
        pass

    def miditer(self, PHiter, conv):
        if (self.rank == 0):
            print('{itr:3d} {conv:12.4e}'.format(itr=PHiter, conv=conv))

    def enditer(self, PHiter):
        pass

    def post_everything(self, PHiter, conv):
        pass
