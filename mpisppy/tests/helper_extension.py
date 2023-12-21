# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# This is an extension to be used for testing.

from collections import defaultdict
import mpisppy.utils.sputils as sputils
import mpisppy.extensions.xhatbase
import mpisppy.phbase as phbase

class TestHelperExtension(mpisppy.extensions.extension.Extension):
    """
    Args:
        spo (SPOpt object): the calling object
    """
    def __init__(self, spo):
        super().__init__(spo)
        self.checked = defaultdict(list)   # for testing xhatter
        # make it a little easier to find out who has been sent here
        self.opt._TestHelperExtension_checked = self.checked

        
    def pre_solve(self, subproblem):
        pass

        
    def post_solve_loop(self):
        pass

        
    def post_solve(self, subproblem, results):
        self.checked[subproblem.name].append(results.Problem[0].Upper_bound)
        return results

        
    def pre_iter0(self):
        pass

        
    def post_iter0(self):
        pass


    def post_iter0_after_sync(self):
        pass
        
        
    def miditer(self):
        pass

    
    def enditer(self):
        pass

        
    def enditer_after_sync(self):        
        pass
        
        
    def post_everything(self):
        pass

