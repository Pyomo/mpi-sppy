# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import pyomo.environ as pyo

# NOTE: a caller attaches the comms (e.g. pre_iter0)

import mpisppy.extensions.extension
from mpisppy.utils.sputils import _ScenTree


class XhatBase(mpisppy.extensions.extension.PHExtension):
    """
        Any inherited class must implement the preiter0, postiter etc. methods
        
        Args:
            opt (SPBase object): gives the problem that we bound

        Attributes:
          scenario_name_to_rank (dict of dict): nodes (i.e. comms) scen names
                keys are comms (i.e., tree nodes); values are dicts with keys
                that are scenario names and values that are ranks
    """
    def __init__(self, opt):
        super().__init__(opt)
        self.rank = self.opt.rank
        self.n_proc = self.opt.n_proc
        self.verbose = self.opt.options["verbose"]

        scen_count = len(opt.all_scenario_names)

        self.scenario_name_to_rank = opt.scenario_names_to_rank
        # dict: scenario names --> LOCAL rank number (needed mainly for xhat)
        
     #**********
    def _try_one(self, snamedict, solver_options=None, verbose=False, restore_nonants=True):
        """ try the scenario named sname in self.opt.local_scenarios
       Args:
            snamedict (dict): key: scenario tree non-leaf name, val: scen name
                            the scen name can be None if it is not local
            solver_options (dict): passed through to the solver
            verbose (boolean): controls output
            restore_nonants (bool): if True, restores the nonants to their original
                                    values in all scenarios. If False, leaves the
                                    nonants as they are in the tried scenario
       NOTE: The solve loop is with fixed nonants so W and rho do not
             matter to the optimization. When we want to check the obj
             value we need to drop them, but we don't need to re-optimize
             since all other Vars are optimized already with the nonants
             fixed.
        Returns:
             obj (float or None): the expected value for sname as xhat or None
        """
        xhats = dict()  # to pass to _fix_nonants
        self.opt._save_nonants()  # (BTW: for all local scenarios)
        # For now, we are going to treat two-stage as a special case
        if len(snamedict) == 1:
            sname = snamedict["ROOT"]  # also serves as an assert
            if sname in self.opt.local_scenarios:
                xhat = self.opt.local_scenarios[sname]._PySP_nonant_cache
            else:
                xhat = None
            src_rank = self.scenario_name_to_rank["ROOT"][sname]
            try:
                xhats["ROOT"] = self.comms["ROOT"].bcast(xhat, root=src_rank)
            except:
                print("rank=",self.rank, "xhats bcast failed on src_rank={}"\
                      .format(src_rank))
                print("root comm size={}".format(self.comms["ROOT"].size))
                raise
        else:  # multi-stage
            # assemble parts and put it in xhats
            # send to ranks in the comm or receive ANY_SOURCE
            # (for closest do allreduce with loc operator) rank
            nlens = dict()
            cistart = dict()  # ci start for each local node
            for k, s in self.opt.local_scenarios.items():
                for nnode in s._PySPnode_list:
                    ndn = nnode.name
                    if ndn not in cistart:
                        cistart[ndn] = s._PySP_cistart[ndn]
                    if ndn not in nlens:
                        nlens[ndn] = s._PySP_nlens[ndn]
                    if ndn not in xhats:
                        xhats[ndn] = None
                    if ndn not in snamedict:
                        raise RuntimeError(f"{ndn} not in snamedict={snamedict}")
                    if snamedict[ndn] == k:
                        # cache lists are just concated node lists
                        xhats[ndn] = [s._PySP_nonant_cache[i+cistart[ndn]]
                                      for i in range(nlens[ndn])]
            for ndn in cistart:  # local nodes
                if snamedict[ndn] not in self.scenario_name_to_rank[ndn]:
                    print (f"For ndn={ndn}, snamedict[ndn] not in "
                           "self.scenario_name_to_rank[ndn]")
                    print(f"snamedict[ndn]={snamedict[ndn]}")
                    print(f"self.scenario_name_to_rank[ndn]={self.scenario_name_to_rank[ndn]}")
                    raise RuntimeError("Bad scenario selection for xhat")
                src_rank = self.scenario_name_to_rank[ndn][snamedict[ndn]]
                try:
                    xhats[ndn] = self.comms[ndn].bcast(xhats[ndn], root=src_rank)
                except:
                    print("rank=",self.rank, "xhats bcast failed on ndn={}, src_rank={}"\
                          .format(ndn,src_rank))
                    raise
    
            # assemble xhat (which is a nonants dict) from xhats
            for ndn in xhats:
                for i in range(cistart[ndn], nlens[ndn]):
                    xhats[ndn] = xhats[ndn]
        # The save is done above
        self.opt._fix_nonants(xhats)  # (BTW: for all local scenarios)

        # Special Tee option for xhat
        sopt = solver_options
        Tee=False
        if solver_options is not None and "Tee" in solver_options:
            sopt = dict(solver_options)
            Tee = sopt["Tee"]
            del sopt["Tee"]

        # NOTE: for APH we may need disable_pyomo_signal_handling
        self.opt.solve_loop(solver_options=sopt,
                           dis_W=True, dis_prox=True,
                           verbose=verbose,
                           tee=Tee)

        infeasP = self.opt.infeas_prob()
        if infeasP != 0.:
            # restoring does no harm
            # if this solution is infeasible
            self.opt._restore_nonants()
            return None
        else:
            if verbose and src_rank == self.rank:
                print("   Feasible xhat found:")
                self.opt.local_scenarios[sname].pprint()
            self.opt._disable_W_and_prox()
            obj = self.opt.Eobjective(verbose=verbose)
            if restore_nonants:
                self.opt._restore_nonants()
                self.opt._reenable_W_and_prox()  # not needed when a spoke
            return obj

    #**********
    def csv_nonants(self, snamedict, fname):
        """ write the non-ants in csv format to files based on the file name
            (we will over-write files if they already exists)
            Args:
               snamedic (str): the names of the scenarios to use
               fname (str): the full name of the file to which to write
        """
        # only the rank with the requested scenario writes
        for ndn, sname in snamedict.items():
            if sname not in self.opt.local_scenarios:
                continue
            scen = self.opt.local_scenarios[sname]
            with open(fname+"_"+ndn+"_"+sname+".csv", "w") as f:
                for node in scen._PySPnode_list:
                    if node.name == ndn:
                        break
                nlens = scen._PySP_nlens
                f.write(ndn)
                for i in range(nlens[ndn]):
                    vardata = node.nonant_vardata_list[i]
                    f.write(', "'+vardata.name+'", '+str(vardata._value))
                f.write("\n")

    #**********
    def csv_allvars(self, snamedict, fname):
        """ write all Vars in csv format to files based on the file name
            (we will over-write files if they already exists)
            Args:
               snamedict (dict): scenario names
               fname (str): the full name of the file to which to write
        """
        # only the rank with the requested scenario writes
        for ndn, sname in snamedict.items():
            if sname not in self.opt.local_scenarios:
                continue
            scen = self.opt.local_scenarios[sname]
            for node in scen._PySPnode_list:
                if node.name == ndn:
                    break
                with open(fname+"_"+ndn+"_"+sname,"w") as f:
                    for ((v_name, v_index), v_data)\
                        in scen.component_data_iterindex(pyo.Var, active=True):
                        f.write(v_name + ", " + str(pyo.value(v_data)) + "\n")

    """ Functions to be called by xhat extensions. This is just code
    common to all, or almost all xhat extensions. It happens to be the
    case that whatever extobject that gets passed in will have been
    derived from XhatBase, but that is not why the code is here. It was
    factored simply for the usual reasons to factor code.  """

    def xhat_common_post_everything(self, extname, obj, snamedict):
        """ Code that most xhat post_everything routines will want to call.
        Args:
            extname (str): the name of the extension for reporting
            obj (float): the xhat objective function
            snamedict (dict): the (scenario) names upon which xhat is based
        """
        if self.rank == 0 and self.verbose:
            print ("****", extname ,"Used scenarios",
                   str(snamedict),"to get xhat Eobj=",obj)

        if "csvname" in self.options:
            self.csv_nonants(snamedict, self.options["csvname"])

        if "dump_prefix" in self.options:
            prefpref = self.options["dump_prefix"]
            pref = extname
            self.csv_nonants(snamedict, prefpref + "_nonant_" + pref)
            self.csv_allvars(snamedict, prefpref + "_allvars_" + pref)

    def post_iter0(self):
        # the base class needs this
        self.comms = self.opt.comms
        
"""
May 2020, DLW
Simple tree with branching factors (arbitrary trees later)

==============

Background:

0. We have a scenario tree defined by a ROOT and branching factors for
each stage beyond the first stage. E.g., for a three stage tree the
branching factors might be [2,3].We refer to the final stage tree
nodes and "leaf" nodes and all other nodes as "non-leaf" nodes. In the
example, there are three non-leaf nodes: ROOT, plus the two nodes at
in the second stage.

1. One path through the tree defines a scenario, so the number of
scenarios is the product of the branching factors.

2. We want to allow an arbitrary number of stages but, realistically, there would be at most 6.

3. Processing is done on ranks and scenarios must be assigned to ranks (MPI terminology).

4. We want as much flexibility as possible when determining the number of ranks, but
it might not be unlimited flexibility.

5. For a given rank, the scenarios assigned to it must all be in the same non-leaf nodes. The
reasons have to do with how we will use MPI comms: There will be a comm for each non-leaf node
and reductions will be done for the node on its corresponding comm. This is critical.

=================

Input: number of ranks, list of branching factors, list of scenario names

output: An error message concerning the relationship between the inputs
or

A dictionary mapping each non-leaf node name in the scenario tree
to a dictionary mapping scenario name to rank number.

(There is redundancy in the inputs and outputs, but both are handy for use.)

=================

Naming conventions:

= non-leaf nodes other than ROOT are named <parent>_nodenum so for the
example, there are three non-leaf nodes named ROOT, ROOT_1, and ROOT_2

= we happen to be given a list of scenario names that has the same
length as the number of scenarios (i.e., the product of the branching
factors) to recover (or construct) the scenario names, the tree is
traversed depth-first.  As a practical matter, they will have names
like Scenx or Scenariox where x is replaced by a serial number.

==================

Notes:

The outer-most non-leaf nodes completely determine the assignment of scenarios
to ranks.

So maybe we should flip it around and assign the ranks to the outer-most
non-leaf tree nodes.

A rank cannot span two such nodes, but a node can have two ranks.
"""
