# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# PH-specific code

import mpisppy.phbase
import shutil
import mpisppy.MPI as mpi

# decorator snarfed from stack overflow - allows per-rank profile output file generation.
def profile(filename=None, comm=mpi.COMM_WORLD):
    pass

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()


############################################################################
class PH(mpisppy.phbase.PHBase):
    """ PH. See PHBase for list of args. """

    #======================================================================
    # uncomment the line below to get per-rank profile outputs, which can 
    # be examined with snakeviz (or your favorite profile output analyzer)
    #@profile(filename="profile_out")
    def ph_main(self, finalize=True):
        """ Execute the PH algorithm.

        Args:
            finalize (bool, optional, default=True):
                If True, call `PH.post_loops()`, if False, do not,
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
                trivial_bound (float):
                    The "trivial bound", computed by solving the model with no
                    nonanticipativity constraints (immediately after iter 0).

        NOTE:
            You need an xhat finder either in denoument or in an extension.
        """
        verbose = self.options['verbose']
        self.PH_Prep()
        # Why is subproblem_creation() not called in PH_Prep? Answer: xhat_eval.
        self.subproblem_creation(verbose)

        if (verbose):
            print('Calling PH Iter0 on global rank {}'.format(global_rank))
        trivial_bound = self.Iter0()
        if (verbose):
            print ('Completed PH Iter0 on global rank {}'.format(global_rank))
        if ('asynchronousPH' in self.options) and (self.options['asynchronousPH']):
            raise RuntimeError("asynchronousPH is deprecated; use APH")

        self.iterk_loop()

        if finalize:
            Eobj = self.post_loops(self.extensions)
        else:
            Eobj = None

        return self.conv, Eobj, trivial_bound


if __name__ == "__main__":
    #==============================
    # hardwired by dlw for debugging
    import mpisppy.tests.examples.farmer as refmodel
    import mpisppy.utils.sputils as sputils

    PHopt = {}
    PHopt["asynchronousPH"] = False
    PHopt["solvername"] = "cplex"
    PHopt["PHIterLimit"] = 5
    PHopt["defaultPHrho"] = 1
    PHopt["convthresh"] = 0.001
    PHopt["verbose"] = True
    PHopt["display_timing"] = True
    PHopt["display_progress"] = True
    # one way to set up options (never mind that this is not a MIP)
    PHopt["iter0_solver_options"]\
        = sputils.option_string_to_dict("mipgap=0.01")
    # another way
    PHopt["iterk_solver_options"] = {"mipgap": 0.001}

    ScenCount = 50
    all_scenario_names = ['scen' + str(i) for i in range(ScenCount)]
    # end hardwire

    scenario_creator = refmodel.scenario_creator
    scenario_denouement = refmodel.scenario_denouement

    # now test extensions and convergers together.
    # NOTE that the options is a reference,
    #   so you could have changed PHopt instead.
    PHopt["PHIterLimit"] = 5 # the converger will probably stop it.
    PHopt["asynchronousPH"] = False

    from mpisppy.extensions.xhatlooper import XhatLooper
    from mpisppy.convergers.fracintsnotconv import FractionalConverger
    PHopt["xhat_looper_options"] =  {"xhat_solver_options":\
                                     PHopt["iterk_solver_options"],
                                     "scen_limit": 3,
                                     "csvname": "looper.csv"}
    ph = PH(PHopt, all_scenario_names, scenario_creator, scenario_denouement,
            extensions=XhatLooper,
            ph_converger=FractionalConverger,
            rho_setter=None)
    conv, obj, bnd = ph.ph_main()
    print ("Quitting Early.")
    quit()

    from mpisppy.extensions.diagnoser import Diagnoser
    # now test whatever is new
    ph = PH(PHopt, all_scenario_names, scenario_creator, scenario_denouement,
            extensions=Diagnoser, 
            ph_converger=None,
            rho_setter=None)
    ph.options["PHIterLimit"] = 3

    if global_rank == 0:
        try:
            shutil.rmtree("delme_diagdir")
            print ("...deleted delme_diagdir")
        except:
            pass
    ph.options["diagnoser_options"] = {"diagnoser_outdir": "delme_diagdir"}
    conv, obj, bnd = ph.ph_main()

    import mpisppy.extensions.avgminmaxer as minmax_extension
    from mpisppy.extensions.avgminmaxer import MinMaxAvg
    ph = PH(PHopt, all_scenario_names, scenario_creator, scenario_denouement,
            extensions=MinMaxAvg,
            ph_converger=None,
            rho_setter=None)
    ph.options["avgminmax_name"] =  "FirstStageCost"
    ph.options["PHIterLimit"] = 3
    conv, obj, bnd = ph.ph_main()

        # trying asynchronous operation
    PHopt["asynchronousPH"] = True
    PHopt["async_frac_needed"] = 0.5
    PHopt["async_sleep_secs"] = 1
    ph = PH(PHopt, all_scenario_names, scenario_creator, scenario_denouement)

    conv, obj, bnd = ph.ph_main()

    if global_rank == 0:
        print ("E[obj] for converged solution (probably NOT non-anticipative)",
               obj)

    dopts = sputils.option_string_to_dict("mipgap=0.001")
    objbound = ph.post_solve_bound(solver_options=dopts, verbose=False)
    if global_rank == 0:
        print ("**** Lagrangian objective function bound=",objbound)
        print ("(probably converged way too early, BTW)")
   
