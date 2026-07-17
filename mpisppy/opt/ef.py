###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import mpisppy.spbase
import pyomo.environ as pyo
import logging
import mpisppy.utils.sputils as sputils
import pathlib
import os
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect

logger = logging.getLogger("mpisppy.ef")

class ExtensiveForm(mpisppy.spbase.SPBase):
    """ Create and solve an extensive form. 

    Attributes:
        ef (:class:`pyomo.environ.ConcreteModel`):
            Pyomo model of the extensive form.
        solver:
            Solver produced by the Pyomo solver factory.

    Args:
        options (dict):
            Dictionary of options. May include a `solver` key to
            specify which solver name to use on the EF.
        all_scenario_names (list):
            List of the names of each scenario in the EF (strings).
        scenario_creator (callable):
            Scenario creator function, which takes as input a scenario
            name, and returns a Pyomo model of that scenario.
        scenario_creator_kwargs (dict, optional):
            Keyword arguments passed to `scenario_creator`.
        all_nodenames (list, optional):
            List of all node names, incl. leaves. Can be None for two-stage
            problem.
        model_name (str, optional):
            Name of the resulting EF model object.
        suppress_warnings (bool, optional):
            Boolean to suppress warnings when building the EF. Default
            is False.

    Note: allowing use of the "solver" option key is for backward compatibility

    """
    def __init__(
        self,
        options,
        all_scenario_names,
        scenario_creator,
        scenario_creator_kwargs=None,
        all_nodenames=None,
        model_name=None,
        suppress_warnings=False,
        extensions=None,
        extension_kwargs=None,
        mutable_probability=None,
    ):
        """ Create the EF and associated solver. """
        super().__init__(
            options,
            all_scenario_names,
            scenario_creator,
            scenario_creator_kwargs=scenario_creator_kwargs,
            all_nodenames=all_nodenames
        )        

        self.bundling = True
        if self.n_proc > 1 and self.cylinder_rank == 0:
            logger.warning("Creating an ExtensiveForm object in parallel. Why?")
        required = ["solver"]
        self._options_check(required, self.options)
        self.solver = pyo.SolverFactory(self.options["solver"])

        # When True, scenario probabilities are stored as mutable Pyomo Params
        # so they can be updated in place (see set_scenario_probabilities).
        # Falls back to the "mutable_probability" option key for convenience.
        if mutable_probability is None:
            mutable_probability = self.options.get("mutable_probability", False)
        self.mutable_probability = mutable_probability
        # Tracks whether a persistent solver already has this EF loaded, so
        # solve_extensive_form(reuse_instance=True) can skip set_instance.
        self._instance_loaded = False

        self.extensions = extensions
        self.extension_kwargs = extension_kwargs
        
        if (self.extensions is not None):
            if self.extension_kwargs is None:
                self.extobject = self.extensions(self)
            else:
                self.extobject = self.extensions(
                    self, **self.extension_kwargs
                )

        if self.options.get("solver_log_dir", None):
            if self.global_rank == 0:
                # create the directory if not there
                directory = self.options["solver_log_dir"]
                try:
                    pathlib.Path(directory).mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    raise FileExistsError(f"solver-log-dir={directory} already exists!")
        
        self.ef = sputils._create_EF_from_scen_dict(self.local_scenarios,
                EF_name=model_name,
                mutable_probability=self.mutable_probability)

    def solve_extensive_form(self, solver_options=None, tee=False,
                             reuse_instance=False):
        """ Solve the extensive form.

            Args:
                solver_options (dict, optional):
                    Dictionary of solver-specific options (e.g. Gurobi options,
                    CPLEX options, etc.).
                tee (bool, optional):
                    If True, displays solver output. Default False.
                reuse_instance (bool, optional):
                    If True and this EF has already been loaded into a
                    persistent solver, skip re-loading the instance
                    (``set_instance``) and re-solve the already-loaded model.
                    Use this to re-solve cheaply after updating mutable data
                    such as scenario probabilities (see
                    set_scenario_probabilities). Ignored for non-persistent
                    solvers. Default False.

            Returns:
                :class:`pyomo.opt.results.results_.SolverResults`:
                    Result returned by the Pyomo solve method.

        """
        # Recognizes both legacy PersistentSolver and APPSI /
        # pyomo.contrib.solver interfaces (e.g. appsi_highs); see
        # sputils.has_persistent_solve_api for why this is not is_persistent.
        persistent = sputils.has_persistent_solve_api(self.solver)
        if persistent and not (reuse_instance and self._instance_loaded):
            self.solver.set_instance(self.ef)
            self._instance_loaded = True


        solve_keyword_args = dict()            
        if self.options.get("solver_log_dir", None):            
            # solver-log logic copied from spopt.py
            dir_name = self.options["solver_log_dir"]
            file_name = "EF_solver_log.log"
            # Workaround for Pyomo/pyomo#3589: Setting 'keepfiles' to True is required
            # for proper functionality when using the GurobiDirect / GurobiPersistent solver.
            if isinstance(self.solver, GurobiDirect):
                if solver_options is None:
                    solver_options = dict()
                solver_options["LogFile"] = os.path.join(dir_name, file_name)
            else:
                solve_keyword_args["logfile"] = os.path.join(dir_name, file_name)
            
        # Pass solver-specifiec (e.g. Gurobi, CPLEX) options
        if solver_options is not None:
            for (opt, value) in solver_options.items():
                self.solver.options[opt] = value
                
        results = self.solver.solve(self.ef, tee=tee, load_solutions=False, **solve_keyword_args)
        if sputils.not_good_enough_results(results):
            # this should catch infeasible and unbounded cases
            return results
        
        if persistent:
            self.solver.load_vars()
        else:
            self.ef.solutions.load_from(results)

        self.first_stage_solution_available = True
        self.tree_solution_available = True

        return results

    def set_scenario_probabilities(self, prob_map):
        """ Update scenario probabilities on a mutable-probability EF in place.

            Requires the EF to have been created with
            ``mutable_probability=True``. The supplied probabilities replace the
            current values; the full set of scenario probabilities must sum to
            1 after the update (this is a full EF). If a persistent solver has
            already been loaded, the objective is re-pushed so a subsequent
            ``solve_extensive_form(reuse_instance=True)`` re-solves with the new
            probabilities without rebuilding the model.

            Args:
                prob_map (dict):
                    Maps scenario name to its new probability. Names not present
                    are left unchanged.

            Raises:
                RuntimeError:
                    If the EF was not built with ``mutable_probability=True``.
                KeyError:
                    If ``prob_map`` contains an unknown scenario name.
                ValueError:
                    If the resulting probabilities do not sum to 1.
        """
        if not self.mutable_probability:
            raise RuntimeError(
                "set_scenario_probabilities requires the ExtensiveForm to be "
                "created with mutable_probability=True.")
        prob = self.ef._mpisppy_model.prob
        # Validate before applying so a bad call leaves the model unchanged.
        # Unmentioned scenarios keep their current probability (partial updates
        # are allowed as long as the resulting full vector still sums to 1).
        for sname in prob_map:
            if sname not in prob:
                raise KeyError(f"Unknown scenario name '{sname}' in prob_map.")
        resulting = {sn: prob_map.get(sn, pyo.value(prob[sn]))
                     for sn in self.ef._ef_scenario_names}
        total = sum(resulting.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"scenario probabilities must sum to 1; got {total}.")
        for sname, p in prob_map.items():
            prob[sname].value = p
            # keep _mpisppy_probability consistent for downstream readers
            getattr(self.ef, sname)._mpisppy_probability = p
        # Re-push the objective so a persistent solver picks up the new
        # coefficients. Required for legacy persistent solvers; harmless (and
        # redundant with auto-tracking) for APPSI / pyomo.contrib.solver.
        if self._instance_loaded and \
                sputils.has_persistent_solve_api(self.solver):
            self.solver.set_objective(self.ef.EF_Obj)

    def get_objective_value(self):
        """ Retrieve the objective value.
        
        Returns:
            float:
                Objective value.

        Raises:
            ValueError:
                If optimal objective value could not be retrieved.
        """
        if not self.tree_solution_available:
            obj_val = None
        else:        
            try:
                obj_val = pyo.value(self.ef.EF_Obj)
            except Exception as e:
                raise ValueError(f"Could not extract EF objective value with error: {str(e)}")

        if (self.extensions is not None):
            obj_val = self.extobject.get_objective_value(obj_val)
        
        return obj_val

    def get_root_solution(self):
        """ Get the value of the variables at the root node.

        Returns:
            dict:
                Dictionary mapping variable name (str) to variable value
                (float) for all variables at the root node.
        """
        result = dict()
        for var in self.ef.ref_vars.values():
            var_name = var.name
            dot_index = var_name.find(".")
            if dot_index >= 0 and var_name[:dot_index] in self.all_scenario_names:
                var_name = var_name[dot_index+1:]
            result[var_name] = var.value
        return result

    def nonants(self):
        """ An iterator to give representative Vars subject to non-anticipitivity
        Args: None

        Yields:
            tree node name, full EF Var name, Var value
        """
        yield from sputils.ef_nonants(self.ef)


    def nonants_to_csv(self, filename):
        """ Dump the nonant vars from an ef to a csv file; truly a dump...
        Args:
            filename (str): the full name of the csv output file
        """
        sputils.ef_nonants_csv(self.ef, filename)


    def scenarios(self):
        """ An iterator to give the scenario sub-models in an ef
        Args: None

        Yields:
            scenario name, scenario instance (str, ConcreteModel)
        """
        yield from self.local_scenarios.items()


if __name__ == "__main__":
    # for ad hoc developer testing
    import mpisppy.tests.examples.farmer as farmer

    """ Farmer example """
    scenario_names = ["Scen" + str(i) for i in range(3)]
    scenario_creator_kwargs = {"sense": pyo.minimize, "use_integer": False}
    options = {"solver": "gurobi"}
    ef = ExtensiveForm(
        options,
        scenario_names,
        farmer.scenario_creator,
        model_name="TestEF",
        scenario_creator_kwargs=scenario_creator_kwargs,
    )
    results = ef.solve_extensive_form()
    print("Farmer objective value:", pyo.value(ef.ef.EF_Obj))
