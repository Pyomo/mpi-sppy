###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""ADMM wrapper for --admm; shared helpers for the stoch-admm family.

This module hosts AdmmWrapper (used by --admm) and helpers that
Stoch_AdmmWrapper and AdmmBundler also import.

----------------------------------------------------------------------
Vocabulary used consistently across the ADMM wrappers and their tests
----------------------------------------------------------------------

before-wrap scenario (a.k.a. before-wrap extended scenario)
    The per-(ADMM subproblem, stochastic scenario) Pyomo model that
    the user's scenario_creator returns.  Has the user's scenario
    tree (stages 1..T) but no ADMM consensus stage yet.

wrapped scenario (a.k.a. wrapped extended scenario)
    The same Pyomo model after the wrapper applies the wrap
    operation.  This is what mpi-sppy iterates over.  For
    Stoch_AdmmWrapper, the tree gains an ADMM consensus stage at a
    new last node; for AdmmBundler, the bundle carries every
    consensus Var on a single ROOT node.

wrap (verb), narrow scope
    The ADMM-specific transformation between before-wrap and
    wrapped: append the ADMM consensus stage (Stoch_AdmmWrapper) or
    flatten consensus Vars into ROOT (AdmmBundler), plus the
    consequent surrogate-Var / probability bookkeeping.  Wrap does
    NOT cover sputils.attach_root_node -- whether the user calls it
    in scenario_creator or the wrapper calls it on the user's behalf
    via the first_stage_cost / first_stage_varlist hooks, that step
    is setup, not wrap.

the wrapper
    Any of AdmmWrapper, Stoch_AdmmWrapper, AdmmBundler.  Use the
    class name when which wrapper matters.

stochastic scenario (paper: xi in Xi)
    One realization of the random data; the user has |Xi| of them.
    Always write "stochastic scenario" in prose -- bare "scenario"
    is ambiguous now that before-wrap / wrapped also exist.

ADMM subproblem (paper: a in A)
    One partition / region of the decomposed problem; the user has
    |A| of them.  ADMM capitalized in prose; the code identifier is
    admm_subproblem.  scenario_creator is called once per
    (ADMM subproblem, stochastic scenario) pair, ordering matched to
    the ADMM_STOCH_{admm}_{stoch} naming convention.

first-stage Vars / first-stage cost
    The Vars at stage 1 of the before-wrap scenario tree, and the
    corresponding cost expression.  Paper-aligned, algorithm-level
    term; use it when describing the user-facing API
    (first_stage_varlist / first_stage_cost hooks) or the algorithm
    in the abstract.

root-node Vars
    The Vars attached to _mpisppy_node_list[0].  In the standard
    case these are the user's first-stage Vars after
    sputils.attach_root_node has run.  Use this term when the
    surrounding prose is about what mpi-sppy is doing to the
    node-list data structure.
"""

import mpisppy.utils.sputils as sputils
import pyomo.environ as pyo
from mpisppy import MPI
global_rank = MPI.COMM_WORLD.Get_rank()

def _admm_normalize_consensus_vars(consensus_vars, *, tuple_form):
    """Coerce every Var identifier in consensus_vars to its .name string.

    Each entry of a subproblem's consensus list may be a string
    (legacy) or a Pyomo Var / VarData (anything exposing a .name
    attribute).  Mixed lists are allowed so callers can migrate one
    entry at a time.  For indexed Vars, pass individual VarData
    objects (e.g., scenario.x[idx]) rather than the container.

    Caveat: a Pyomo VarData holds its parent block via a weakref, so
    the .name lookup here only resolves to a real name if the
    before-wrap scenario the Var was taken from is still alive when
    the wrapper is constructed.  Callers that build a before-wrap
    scenario inside a helper function must keep it alive across the
    wrapper call, or snapshot the names themselves before letting it
    go out of scope.

    Args:
        consensus_vars (dict): {ADMM subproblem name: list_of_entries}.
            tuple_form=False (--admm): entries are Var identifiers.
            tuple_form=True  (--stoch-admm): entries are
                (Var identifier, stage) tuples.
        tuple_form (bool): which list shape consensus_vars uses.

    Returns:
        dict: a new dict with the same shape, all identifiers as
        strings.
    """
    def _to_name(v):
        if isinstance(v, str):
            return v
        name = getattr(v, "name", None)
        if name is None:
            raise TypeError(
                f"consensus_vars entry must be a string or a Pyomo "
                f"Var/VarData (anything with a .name attribute), got "
                f"{type(v).__name__}: {v!r}"
            )
        return name

    out = {}
    for sub, entries in consensus_vars.items():
        if tuple_form:
            out[sub] = [(_to_name(v), stage) for (v, stage) in entries]
        else:
            out[sub] = [_to_name(v) for v in entries]
    return out


def _merge_first_stage_into_consensus_vars(consensus_vars, first_stage_var_names_per_sub, root_stage=1):
    """Add each ADMM subproblem's first-stage Var names to its
    consensus_vars entry, tagged at root_stage.

    Used by --stoch-admm at wrapper-construction time (NOT during
    wrap) when the first_stage_cost / first_stage_varlist hooks are
    defined, so the user's consensus_vars_creator can return only the
    admm-consensus Vars and leave first-stage Vars to the wrapper.

    Per-subproblem because different ADMM subproblems may carry
    different first-stage Vars (e.g., one region owns its factory
    production decisions, a different region owns its own).  Callers
    gather the names by invoking first_stage_varlist on one
    before-wrap scenario per ADMM subproblem.  Those names will end
    up on the root node of every wrapped scenario in the subproblem,
    because Stoch_AdmmWrapper later runs sputils.attach_root_node
    over the same first_stage_varlist; this helper is what makes the
    merge of first-stage Vars into the cross-subproblem consensus
    list happen automatically.

    De-duplicates: entries already present (e.g., from a partially
    migrated consensus_vars_creator that still pre-merges
    first-stage Vars by hand) are left in place.

    Args:
        consensus_vars (dict): {ADMM subproblem name: [(name_str, stage),
            ...]}, already normalized by _admm_normalize_consensus_vars.
        first_stage_var_names_per_sub (dict): {ADMM subproblem name:
            [name_str, ...]}.  ADMM subproblems absent from this map
            get no merge.
        root_stage (int): stage tag for the appended entries (1 for the
            user's stage 1, i.e. the root of the before-wrap scenario
            tree in a 2-stage-origin problem).

    Returns:
        dict: a new consensus_vars dict with first-stage entries merged in.
    """
    out = {}
    for sub, entries in consensus_vars.items():
        existing = set(entries)
        merged = list(entries)
        for name in first_stage_var_names_per_sub.get(sub, []):
            key = (name, root_stage)
            if key not in existing:
                merged.append(key)
                existing.add(key)
        out[sub] = merged
    return out


def _consensus_vars_number_creator(consensus_vars):
    """associates to each consensus vars the number of time it appears

    Args:
        consensus_vars (dict): dictionary which keys are the subproblems and values are the list of consensus variables 
        present in the subproblem

    Returns:
        consensus_vars_number (dict): dictionary whose keys are the consensus variables 
        and values are the number of subproblems the variable is linked to.
    """
    consensus_vars_number={}
    for subproblem in consensus_vars:
        for var in consensus_vars[subproblem]:
            if var not in consensus_vars_number: # instanciates consensus_vars_number[var]
                consensus_vars_number[var] = 0
            consensus_vars_number[var] += 1
    for var in consensus_vars_number:
        if consensus_vars_number[var] == 1:
            print(f"The consensus variable {var} appears in a single subproblem")
    return consensus_vars_number

class AdmmWrapper():
    """ This class assigns variable probabilities and creates wrapper for the scenario creator

        Args:
            options (dict): options
            all_scenario_names (list): all scenario names
            scenario_creator (fct): returns a concrete model with special things
            consensus_vars (dict): dictionary which keys are the subproblems and values are the list of consensus variables 
            present in the subproblem
            n_cylinder (int): number of cylinders that will ultimately be used
            mpicomm (MPI comm): creates communication
            scenario_creator_kwargs (dict): kwargs passed directly to scenario_creator.
            verbose (boolean): if True gives extra debugging information

        Attributes:
          local_scenarios (dict of scenario objects): concrete models with 
                extra data, key is name
          local_scenario_names (list): names of locals 
    """
    def __init__(self,
            options,
            all_scenario_names,
            scenario_creator, #supplied by the user/ modeller, used only here
            consensus_vars,
            n_cylinders,
            mpicomm,
            scenario_creator_kwargs=None,
            verbose=None,
    ):
        assert len(options) == 0, "no options supported by AdmmWrapper"
        # We need local_scenarios
        self.local_scenarios = {}
        scenario_tree = sputils._ScenTree(["ROOT"], all_scenario_names)
        assert mpicomm.Get_size() % n_cylinders == 0, \
            f"{mpicomm.Get_size()=} and {n_cylinders=}, but {mpicomm.Get_size() % n_cylinders=} should be 0"
        ranks_per_cylinder = mpicomm.Get_size() // n_cylinders 
        
        scenario_names_to_rank, _rank_slices, _scenario_slices =\
                scenario_tree.scen_names_to_ranks(ranks_per_cylinder)

        self.cylinder_rank = mpicomm.Get_rank() // n_cylinders
        #self.cylinder_rank = mpicomm.Get_rank() % ranks_per_cylinder
        self.all_scenario_names = all_scenario_names
        #taken from spbase
        self.local_scenario_names = [
            all_scenario_names[i] for i in _rank_slices[self.cylinder_rank]
        ]
        for sname in self.local_scenario_names:
            s = scenario_creator(sname, **scenario_creator_kwargs)
            self.local_scenarios[sname] = s
        #we are not collecting instantiation time

        self.consensus_vars = _admm_normalize_consensus_vars(consensus_vars, tuple_form=False)
        self.verbose = verbose
        self.consensus_vars_number = _consensus_vars_number_creator(self.consensus_vars)
        #check_consensus_vars(consensus_vars)
        self.assign_variable_probs(verbose=self.verbose)
        self.number_of_scenario = len(all_scenario_names)

    def var_prob_list(self, s):
        """Associates probabilities to variables and raises exceptions if the model doesn't match the dictionary consensus_vars

        Args:
            s (Pyomo ConcreteModel): scenario

        Returns:
            list: list of pairs (variables id (int), probabilities (float)). The order of variables is invariant with the scenarios.
                If the consensus variable is present in the scenario it is associated with a probability 1/#subproblem
                where it is present. Otherwise it has a probability 0.
        """
        return self.varprob_dict[s]
    
    def assign_variable_probs(self, verbose=False):
        self.varprob_dict = {}

        #we collect the consensus variables
        all_consensus_vars = {var_stage_tuple: None for admm_subproblem_names in self.consensus_vars for var_stage_tuple in self.consensus_vars[admm_subproblem_names]}
        error_list1 = []
        error_list2 = []
        for sname,s in self.local_scenarios.items():
            if verbose:
                print(f"AdmmWrapper.assign_variable_probs is processing scenario: {sname}")
            varlist = list()
            self.varprob_dict[s] = list()
            for vstr in all_consensus_vars.keys():
                v = s.find_component(vstr)
                if vstr in self.consensus_vars[sname]:
                    if v is not None:
                        #variables that should be on the model
                        self.varprob_dict[s].append((id(v),1/(self.consensus_vars_number[vstr])))
                    else:
                        error_list1.append((sname,vstr))
                else:
                    if v is None: 
                        # This var will not be indexed but that might not matter??
                        # Going to replace the brackets
                        v2str = vstr.replace("[","__").replace("]","__") # To distinguish the consensus_vars fixed at 0
                        v = pyo.Var()
                        
                        ### Lines equivalent to setattr(s, v2str, v) without warning
                        s.del_component(v2str)
                        s.add_component(v2str, v)

                        v.fix(0)  # pylint: disable=no-member
                        self.varprob_dict[s].append((id(v),0))
                    else:
                        error_list2.append((sname,vstr))
                if v is not None: #if None the error is trapped earlier
                    varlist.append(v)
            objfunc = sputils.find_active_objective(s)
            #this will overwrite the nonants already there
            sputils.attach_root_node(s, objfunc, varlist) 

        if len(error_list1) + len(error_list2) > 0:
            raise RuntimeError (f"for each pair (scenario, variable) of the following list, the variable appears"
                                f"in consensus_vars, but not in the model:\n {error_list1} \n"
                                f"for each pair (scenario, variable) of the following list, the variable appears "
                                f"in the model, but not in consensus var: \n {error_list2}")


    def get_scenario_unscaled(self, sname):
        """Return pre-created scenario without objective scaling (for bundling).

        Args:
            sname (str): scenario name

        Returns:
            Pyomo ConcreteModel: the scenario model (not objective-scaled)
        """
        return self.local_scenarios[sname]

    def admmWrapper_scenario_creator(self, sname):
        #this is the function the user will supply for all cylinders 
        assert sname in self.local_scenario_names, f"{global_rank=} {sname=} \n {self.local_scenario_names=}"
        #should probably be deleted as it takes time
        scenario = self.local_scenarios[sname]

        # Grabs the objective function and multiplies its value by the number of scenarios to compensate for the probabilities
        obj = sputils.find_active_objective(scenario)
        obj.expr = obj.expr * self.number_of_scenario

        return scenario
    
