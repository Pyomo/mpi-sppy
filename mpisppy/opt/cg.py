###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import mpisppy.MPI as mpi
import mpisppy.cgbase

# decorator snarfed from stack overflow - allows per-rank profile output file generation.
def profile(filename=None, comm=mpi.COMM_WORLD):
    pass

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()


############################################################################
class CG(mpisppy.cgbase.CGBase):
    """ CG. See CGBase for list of args. """

    #======================================================================
    # uncomment the line below to get per-rank profile outputs, which can
    # be examined with snakeviz (or your favorite profile output analyzer)
    #@profile(filename="profile_out")
    def cg_main(self, finalize=True):
        """ Execute the CG algorithm.

        Args:
            finalize (bool, optional, default=True):
                If True, call `CG.post_loops()`, if False, do not,
                and return None for Eobj

        Returns:
            tuple:
                Tuple containing

                conv (float):
                    The convergence value (not easily interpretable).
                Eobj (float or `None`):
                    If `finalize=True`, this is the expected, weighted
                    objective value with the proximal term included. This value
                    is not directly useful. If `finalize=False`, this value is
                    `None`.

        NOTE:
            You need an xhat finder either in denoument or in an extension.
        """
        verbose = self.options['verbose']
        self.CG_Prep()

        if (verbose):
            print(f'Calling {self.__class__.__name__} Iter0 on global rank {global_rank}')
        self.Iter0()
        if (verbose):
            print(f'Completed {self.__class__.__name__} Iter0 on global rank {global_rank}')
        
        self.iterk_loop()

        if finalize:
            print("Terminating CG on rank", self.cylinder_rank , "now computing Integer MP")
            Eobj = self.post_loops(self.extensions)
        else:
            Eobj = None

        return self.conv, Eobj


if __name__ == "__main__":
    #==============================
    # hardwired by dlw for debugging
    import mpisppy.tests.examples.farmer as refmodel
    import mpisppy.utils.sputils as sputils

    CGopt = {}

    CGopt["solver_name"] = "xpress"
    CGopt["CGIterLimit"] = 10
    CGopt["convthresh"] = 0.001
    CGopt["verbose"] = True
    CGopt["display_timing"] = True
    CGopt["display_progress"] = True
    # one way to set up options (never mind that this is not a MIP)
    CGopt["iter0_solver_options"]\
        = sputils.option_string_to_dict("mipgap=0.01")
    # another way
    CGopt["iterk_solver_options"] = {"mipgap": 0.001}

    ScenCount = 50
    all_scenario_names = ['scen' + str(i) for i in range(ScenCount)]
    # end hardwire

    scenario_creator = refmodel.scenario_creator
    scenario_denouement = refmodel.scenario_denouement

    cg = CG(CGopt, all_scenario_names, scenario_creator, scenario_denouement)
    cg.options["CGIterLimit"] = 10
    conv, obj, bnd = cg.cg_main()


    if global_rank == 0:
        print ("E[obj] for converged solution",
               obj)

    dopts = sputils.option_string_to_dict("mipgap=0.001")
   
