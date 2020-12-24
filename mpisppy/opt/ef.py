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
        cb_data (dict):
            Data passed directly to scenario_creator.
        suppress_warnings (bool, optional):
            Boolean to suppress warnings when building the EF. Default
            is False.
    """
    def __init__(
        self,
        options,
        all_scenario_names,
        scenario_creator,
        cb_data=None,
        all_nodenames=None,
        model_name=None,
        suppress_warnings=False,
    ):
        """ Create the EF and associated solver. """
        super().__init__(
            options,
            all_scenario_names,
            scenario_creator,
            cb_data=cb_data,
            all_nodenames=all_nodenames
        )
        if self.n_proc > 1 and self.rank == self.rank0:
            logger.warning("Creating an ExtensiveForm object in parallel. Why?")
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

if __name__ == "__main__":
    import mpisppy.tests.examples.farmer as farmer

    """ Farmer example """
    scenario_names = ["Scen" + str(i) for i in range(3)]
    cb_data = {"sense": pyo.minimize, "use_integer": False}
    options = {"solver": "gurobi"}
    ef = ExtensiveForm(
        options,
        scenario_names,
        farmer.scenario_creator,
        model_name="TestEF",
        cb_data=cb_data,
    )
    results = ef.solve_extensive_form()
    print("Farmer objective value:", pyo.value(ef.ef.EF_Obj))
