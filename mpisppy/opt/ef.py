# This software is distributed under the 3-clause BSD License.
from mpisppy.utils.sputils import create_EF
import mpisppy.spbase
import pyomo.environ as pyo
import logging

logger = logging.getLogger("mpisppy.ef")

class ExtensiveForm(mpisppy.spbase.SPBase):
    """ Create and solve an extensive form. 
    """
    def __init__(
        self,
        options,
        all_scenario_names,
        scenario_creator,
        model_name=None,
        scenario_creator_options=None,
    ):
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
            logger.warning("WARNING: Creating an ExtensiveForm object in parallel. Why?")
        if scenario_creator_options is None:
            scenario_creator_options = dict()
        required = ["solver"]
        self._options_check(required, self.options)
        self.solver = pyo.SolverFactory(self.options["solver"])
        self.ef = create_EF(
            all_scenario_names,
            scenario_creator,
            creator_options=scenario_creator_options,
            EF_name=model_name
        )

    def solve_extensive_form(self, tee=False):
        results = self.solver.solve(self.ef, tee=tee)
        return results

if __name__ == "__main__":
    import mpisppy.examples.farmer.farmer as farmer
    import mpisppy.examples.netdes.netdes as netdes
    import mpisppy.examples.sslp.sslp as sslp

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


    """ Network design example """
    inst = "network-10-20-L-01"
    num_scen = int(inst.split("-")[-3])
    scenario_names = [f"Scen{i}" for i in range(num_scen)]
    scenario_creator_options = {
        "cb_data": f"{netdes.__file__[:-10]}/data/{inst}.dat"
    }
    options = {"solver": "gurobi"}
    ef = ExtensiveForm(
        options,
        scenario_names,
        netdes.scenario_creator,
        model_name="TestEF",
        scenario_creator_options=scenario_creator_options
    )
    results = ef.solve_extensive_form()
    print("Netdes objective value:", pyo.value(ef.ef.EF_Obj))


    """ SSLP example """
    inst = "sslp_15_45_10"
    scenario_creator = sslp.scenario_creator
    num_scen = int(inst.split("_")[-1])
    scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]
    scenario_creator_options = {
        "cb_data": f"{sslp.__file__[:-8]}/data/{inst}/scenariodata",
    }
    options = {"solver": "gurobi"}
    ef = ExtensiveForm(
        options,
        scenario_names,
        sslp.scenario_creator,
        model_name="TestEF",
        scenario_creator_options=scenario_creator_options,
    )
    results = ef.solve_extensive_form()
    print("SSLP objective value:", pyo.value(ef.ef.EF_Obj))
