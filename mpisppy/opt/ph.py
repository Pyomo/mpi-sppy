# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# PH-specific code

import mpisppy.phbase
import shutil
import mpi4py.MPI as mpi
import mpisppy.utils.listener_util.listener_util as listener_util
    
# decorator snarfed from stack overflow - allows per-rank profile output file generation.
def profile(filename=None, comm=mpi.COMM_WORLD):
    pass

fullcomm = mpi.COMM_WORLD
rank_global = fullcomm.Get_rank()


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
        verbose = self.PHoptions['verbose']
        self.PH_Prep()
        # Why is subproblem_creation() not called in PH_Prep?
        self.subproblem_creation(verbose)

        if (verbose):
            print('Calling PH Iter0 on global rank {}'.format(rank_global))
        trivial_bound = self.Iter0()
        if (verbose):
            print ('Completed PH Iter0 on global rank {}'.format(rank_global))
        asynch = False
        if ('asynchronousPH' in self.PHoptions):
            asynch = self.PHoptions["asynchronousPH"]
        if asynch:
            if rank_global == 0:
                print ("  !?!?! trying asynchronous PH!!! xxx no extensions or converger?????")
            sleep_secs = self.PHoptions["async_sleep_secs"]

            synchronizer = listener_util.Synchronizer(comms = self.comms,
                                                   Lens = self.Lens,
                                                   #work_fct = global_iterk_loop,
                                                   work_fct = self.iterk_loop,
                                                   rank = rank_global,
                                                   sleep_secs = sleep_secs,
                                                   asynch = asynch)
            args = [] ### [ph]
            kwargs = {"synchronizer": synchronizer,
                      "PH_extensions": None,
                      "PH_converger": None}
            synchronizer.run(args, kwargs)
        else:
            self.iterk_loop()

        if finalize:
            Eobj = self.post_loops(self.PH_extensions)
        else:
            Eobj = None

        return self.conv, Eobj, trivial_bound


if __name__ == "__main__":
    #==============================
    # hardwired by dlw for debugging
    import mpisppy.examples.farmer.farmer as refmodel
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
    # NOTE that the PHoptions is a reference,
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
            PH_extensions=XhatLooper,
            PH_converger=FractionalConverger,
            rho_setter=None)
    conv, obj, bnd = ph.ph_main()
    print ("Quitting Early.")
    quit()

    from mpisppy.extensions.diagnoser import Diagnoser
    # now test whatever is new
    ph = PH(PHopt, all_scenario_names, scenario_creator, scenario_denouement,
            PH_extensions=Diagnoser, 
            PH_converger=None,
            rho_setter=None)
    ph.PHoptions["PHIterLimit"] = 3

    if rank_global == 0:
        try:
            shutil.rmtree("delme_diagdir")
            print ("...deleted delme_diagdir")
        except:
            pass
    ph.PHoptions["diagnoser_options"] = {"diagnoser_outdir": "delme_diagdir"}
    import mpisppy.extensions.diagnoser as diagnoser
    conv, obj, bnd = ph.ph_main()

    

    import mpisppy.extensions.avgminmaxer as minmax_extension
    from mpisppy.extensions.avgminmaxer import MinMaxAvg
    ph = PH(PHopt, all_scenario_names, scenario_creator, scenario_denouement,
            PH_extensions=MinMaxAvg,
            PH_converger=None,
            rho_setter=None)
    ph.PHoptions["avgminmax_name"] =  "FirstStageCost"
    ph.PHoptions["PHIterLimit"] = 3
    conv, obj, bnd = ph.ph_main()

        # trying asynchronous operation
    PHopt["asynchronousPH"] = True
    PHopt["async_frac_needed"] = 0.5
    PHopt["async_sleep_secs"] = 1
    ph = PH(PHopt, all_scenario_names, scenario_creator, scenario_denouement)

    conv, obj, bnd = ph.ph_main()

    if rank_global == 0:
        print ("E[obj] for converged solution (probably NOT non-anticipative)",
               obj)

    dopts = sputils.option_string_to_dict("mipgap=0.001")
    objbound = ph.post_solve_bound(solver_options=dopts, verbose=False)
    if rank_global == 0:
        print ("**** Lagrangian objective function bound=",objbound)
        print ("(probably converged way too early, BTW)")
   
