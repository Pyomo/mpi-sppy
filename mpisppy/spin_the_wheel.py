# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

from pyomo.environ import value
from mpisppy import haveMPI, global_toc, MPI

from mpisppy.utils.sputils import (
        first_stage_nonant_writer,
        scenario_tree_solution_writer,
        )

class WheelSpinner:

    def __init__(self, hub_dict, list_of_spoke_dict):
        """ top level for the hub and spoke system
        Args:
            hub_dict(dict): controls hub creation
            list_of_spoke_dict(list dict): controls creation of spokes
    
        Returns:
            spcomm (Hub or Spoke object): the object that did the work (windowless)
            opt_dict (dict): the dictionary that controlled creation for this rank
    
        NOTE: the return is after termination; the objects are provided for query.
    
        """
        if not haveMPI:
            raise RuntimeError("spin_the_wheel called, but cannot import mpi4py")
        self.hub_dict = hub_dict
        self.list_of_spoke_dict = list_of_spoke_dict

        self._ran = False

    def spin(self, comm_world=None):
        return self.run(comm_world=comm_world)

    def run(self, comm_world=None):
        """ top level for the hub and spoke system
        Args:
            comm_world (MPI comm): the world for this hub-spoke system
        """
        if self._ran:
            raise RuntimeError("WheelSpinner can only be run once")

        hub_dict = self.hub_dict
        list_of_spoke_dict = self.list_of_spoke_dict

        # Confirm that the provided dictionaries specifying
        # the hubs and spokes contain the appropriate keys
        if "hub_class" not in hub_dict:
            raise RuntimeError(
                "The hub_dict must contain a 'hub_class' key specifying "
                "the hub class to use"
            )
        if "opt_class" not in hub_dict:
            raise RuntimeError(
                "The hub_dict must contain an 'opt_class' key specifying "
                "the SPBase class to use (e.g. PHBase, etc.)"
            )
        if "hub_kwargs" not in hub_dict:
            hub_dict["hub_kwargs"] = dict()
        if "opt_kwargs" not in hub_dict:
            hub_dict["opt_kwargs"] = dict()
        for spoke_dict in list_of_spoke_dict:
            if "spoke_class" not in spoke_dict:
                raise RuntimeError(
                    "Each spoke_dict must contain a 'spoke_class' key "
                    "specifying the spoke class to use"
                )
            if "opt_class" not in spoke_dict:
                raise RuntimeError(
                    "Each spoke_dict must contain an 'opt_class' key "
                    "specifying the SPBase class to use (e.g. PHBase, etc.)"
                )
            if "spoke_kwargs" not in spoke_dict:
                spoke_dict["spoke_kwargs"] = dict()
            if "opt_kwargs" not in spoke_dict:
                spoke_dict["opt_kwargs"] = dict()
    
        if comm_world is None:
            comm_world = MPI.COMM_WORLD
        n_spokes = len(list_of_spoke_dict)
    
        # Create the necessary communicators
        fullcomm = comm_world
        strata_comm, cylinder_comm = _make_comms(n_spokes, fullcomm=fullcomm)
        strata_rank = strata_comm.Get_rank()
        cylinder_rank = cylinder_comm.Get_rank()
        global_rank = fullcomm.Get_rank()
    
        # Assign hub/spokes to individual ranks
        if strata_rank == 0: # This rank is a hub
            sp_class = hub_dict["hub_class"]
            sp_kwargs = hub_dict["hub_kwargs"]
            opt_class = hub_dict["opt_class"]
            opt_kwargs = hub_dict["opt_kwargs"]
            opt_dict = hub_dict
        else: # This rank is a spoke
            spoke_dict = list_of_spoke_dict[strata_rank - 1]
            sp_class = spoke_dict["spoke_class"]
            sp_kwargs = spoke_dict["spoke_kwargs"]
            opt_class = spoke_dict["opt_class"]
            opt_kwargs = spoke_dict["opt_kwargs"]
            opt_dict = spoke_dict

        # Create the appropriate opt object locally
        opt_kwargs["mpicomm"] = cylinder_comm
        opt = opt_class(**opt_kwargs)
    
        # Create the SPCommunicator object (hub/spoke) with
        # the appropriate SPBase object attached
        if strata_rank == 0: # Hub
            spcomm = sp_class(opt, fullcomm, strata_comm, cylinder_comm,
                              list_of_spoke_dict, **sp_kwargs) 
        else: # Spokes
            spcomm = sp_class(opt, fullcomm, strata_comm, cylinder_comm, **sp_kwargs) 
    
        # Create the windows, run main(), destroy the windows
        spcomm.make_windows()
        if strata_rank == 0:
            spcomm.setup_hub()

        global_toc("Starting spcomm.main()")
        spcomm.main()
        if strata_rank == 0: # If this is the hub
            spcomm.send_terminate()
    
        # Anything that's left to do
        spcomm.finalize()
    
        # to ensure the messages below are True
        cylinder_comm.Barrier()
        global_toc(f"Hub algorithm {opt_class.__name__} complete, waiting for spoke finalization")
        global_toc(f"Spoke {sp_class.__name__} finalized", (cylinder_rank == 0 and strata_rank != 0))
    
        fullcomm.Barrier()
    
        ## give the hub the chance to catch new values
        spcomm.hub_finalize()
    
        spcomm.free_windows()
        global_toc("Windows freed")

        self.spcomm = spcomm
        self.opt_dict = opt_dict
        self.global_rank = global_rank
        self.strata_rank = strata_rank
        self.cylinder_rank = cylinder_rank

        if self.strata_rank == 0:
            self.BestInnerBound = spcomm.BestInnerBound
            self.BestOuterBound = spcomm.BestOuterBound
        else: # the cylinder ranks don't track the inner / outer bounds
            self.BestInnerBound = None
            self.BestOuterBound = None

        self._ran = True

    def on_hub(self):
        if not self._ran:
            raise RuntimeError("Need to call WheelSpinner.run() before finding out.")
        return ("hub_class" in self.opt_dict)

    def write_first_stage_solution(self, solution_file_name,
            first_stage_solution_writer=first_stage_nonant_writer):
        """ Write a solution file, if a solution is available, to the solution_file_name provided
        Args:
            solution_file_name : filename to write the solution to
            first_stage_solution_writer (optional) : custom first stage solution writer function
        """
        if not self._ran:
            raise RuntimeError("Need to call WheelSpinner.run() before querying solutions.")
        winner = self._determine_innerbound_winner()
        if winner:
            self.spcomm.opt.write_first_stage_solution(solution_file_name,first_stage_solution_writer)
    
    def write_tree_solution(self, solution_directory_name,
            scenario_tree_solution_writer=scenario_tree_solution_writer):
        """ Write a tree solution directory, if available, to the solution_directory_name provided
        Args:
            solution_file_name : filename to write the solution to
            scenario_tree_solution_writer (optional) : custom scenario solution writer function
        """
        if not self._ran:
            raise RuntimeError("Need to call WheelSpinner.run() before querying solutions.")
        winner = self._determine_innerbound_winner()
        if winner:
            self.spcomm.opt.write_tree_solution(solution_directory_name,scenario_tree_solution_writer)

    def local_nonant_cache(self):
        """ Returns a dict with non-anticipative values at each local node
            We assume that the optimization has been done before calling this
        """
        if not self._ran:
            raise RuntimeError("Need to call WheelSpinner.run() before querying solutions.")
        local_xhats = dict()
        for k,s in self.spcomm.opt.local_scenarios.items():
            for node in s._mpisppy_node_list:
                if node.name not in local_xhats:
                    local_xhats[node.name] = [
                        value(var) for var in node.nonant_vardata_list]
        return local_xhats

    def _determine_innerbound_winner(self):
        if self.spcomm.global_rank == 0:
            if self.spcomm.last_ib_idx is None:
                best_strata_rank = -1
                global_toc("No incumbent solution available to write!")
            else:
                best_strata_rank = self.spcomm.last_ib_idx
        else:
            best_strata_rank = None
    
        best_strata_rank = self.spcomm.fullcomm.bcast(best_strata_rank, root=0)
        return (self.spcomm.strata_rank == best_strata_rank)

def _make_comms(n_spokes, fullcomm=None):
    """ Create the strata_comm and cylinder_comm for hub/spoke style runs
    """
    if not haveMPI:
        raise RuntimeError("make_comms called, but cannot import mpi4py")
    # Ensure that the proper number of processes have been invoked
    nsp1 = n_spokes + 1 # Add 1 for the hub
    if fullcomm is None:
        fullcomm = MPI.COMM_WORLD
    n_proc = fullcomm.Get_size() 
    if n_proc % nsp1 != 0:
        raise RuntimeError(f"Need a multiple of {nsp1} processes (got {n_proc})")

    # Create the strata_comm and cylinder_comm
    # Cryptic comment: intra is vertical, inter is around the hub
    global_rank = fullcomm.Get_rank()
    strata_comm = fullcomm.Split(key=global_rank, color=global_rank // nsp1)
    cylinder_comm = fullcomm.Split(key=global_rank, color=global_rank % nsp1)
    return strata_comm, cylinder_comm
