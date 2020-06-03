import mpisppy.cylinders.spoke

class FrankWolfeOuterBound(mpisppy.cylinders.spoke.OuterBoundSpoke):

    def main(self):
        self.opt.fwph_main(spcomm=self)

    def opt_callback(self):
        if self.got_kill_signal(): # Checks with hub
            # Kill the process no matter what
            return True

        # Tell the hub about the most recent bound
        self.bound = self.opt._local_bound

        # This will kick down to the normal FWPH
        # convergence checks
        return False

