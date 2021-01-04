# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
from mpisppy.utils.sputils import _create_EF_from_scen_dict
import mpisppy.spbase
import pyomo.environ as pyo
import logging

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
        scenario_creator_options (dict):
            Options to pass to the scenario creator. If this dict
            contains a `cb_data` key, the corresponding values will be
            passed to the PHBase constructor.
        suppress_warnings (bool, optional):
            Boolean to suppress warnings when building the EF. Default
            is False.
    """
    def __init__(
        self,
        options,
        all_scenario_names,
        scenario_creator,
        model_name=None,
        scenario_creator_options=None,
        suppress_warnings=False
    ):
        """ Create the EF and associated solver. """
        cb_data = None
        if scenario_creator_options is not None:
            if "cb_data" in scenario_creator_options:
                cb_data = scenario_creator_options["cb_data"]
        super().__init__(
            options,
            all_scenario_names,
            scenario_creator,
            cb_data=cb_data
        )
        if self.n_proc > 1 and self.rank == self.rank0:
            logger.warning("Creating an ExtensiveForm object in parallel. Why?")
        if scenario_creator_options is None:
            scenario_creator_options = dict()
        required = ["solver"]
        self._options_check(required, self.options)
        self.solver = pyo.SolverFactory(self.options["solver"])
        self.ef = _create_EF_from_scen_dict(self.local_scenarios, EF_name=model_name)

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
        results = self.solver.solve(self.ef, tee=tee)
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
            raise ValueError("Could not extract EF objective value with error: {str(e)}")
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


if __name__ == "__main__":
    import mpisppy.tests.examples.farmer as farmer

    """ Farmer example """
    scenario_names = ["Scen" + str(i) for i in range(3)]
    scenario_creator_options = {
        "cb_data": {"sense": pyo.minimize, "use_integer": False}
    }
    options = {"solver": "gurobi"}
    ef = ExtensiveForm(
        options,
        scenario_names,
        farmer.scenario_creator,
        model_name="TestEF",
        scenario_creator_options=scenario_creator_options,
    )
    results = ef.solve_extensive_form()
    print("Farmer objective value:", pyo.value(ef.ef.EF_Obj))
