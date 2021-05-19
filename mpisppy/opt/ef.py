# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import mpisppy.spbase
import pyomo.environ as pyo
import logging
import mpisppy.utils.sputils as sputils

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
            Dictionary of options. Must include a `solver` key to
            specify which solver to use on the EF.
        all_scenario_names (list):
            List of the names of each scenario in the EF (strings).
        scenario_creator (callable):
            Scenario creator function, which takes as input a scenario
            name, and returns a Pyomo model of that scenario.
        model_name (str, optional):
            Name of the resulting EF model object.
        scenario_creator_kwargs (dict):
            Keyword args passed to `scenario_creator`.
        suppress_warnings (bool, optional):
            Boolean to suppress warnings when building the EF. Default
            is False.
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
        self.ef = sputils._create_EF_from_scen_dict(self.local_scenarios,
                EF_name=model_name)

    def solve_extensive_form(self, solver_options=None, tee=False):
        """ Solve the extensive form.
            
            Args:
                solver_options (dict, optional):
                    Dictionary of solver-specific options (e.g. Gurobi options,
                    CPLEX options, etc.).
                tee (bool, optional):
                    If True, displays solver output. Default False.

            Returns:
                :class:`pyomo.opt.results.results_.SolverResults`:
                    Result returned by the Pyomo solve method.
                
        """
        if "persistent" in self.options["solver"]:
            self.solver.set_instance(self.ef)
        # Pass solver-specifiec (e.g. Gurobi, CPLEX) options
        if solver_options is not None:
            for (opt, value) in solver_options.items():
                self.solver.options[opt] = value
        results = self.solver.solve(self.ef, tee=tee, load_solutions=False)
        if len(results.solution) > 0:
            if sputils.is_persistent(self.solver):
                self.solver.load_vars()
            else:
                self.ef.solutions.load_from(results)
            self.first_stage_solution_available = True
            self.tree_solution_available = True
        return results

    def get_objective_value(self):
        """ Retrieve the objective value.
        
        Returns:
            float:
                Objective value.

        Raises:
            ValueError:
                If optimal objective value could not be retrieved.
        """
        try:
            obj_val = pyo.value(self.ef.EF_Obj)
        except Exception as e:
            raise ValueError(f"Could not extract EF objective value with error: {str(e)}")
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
