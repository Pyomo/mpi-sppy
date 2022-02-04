from typing import Dict, Sequence

import pyomo.environ as pyo

from abstract import abstract_model
import mpisppy.scenario_tree


def scenario_creator(
    name: str,
    data_dicts: Sequence[Dict],
) -> pyo.ConcreteModel:
    abstract = abstract_model()

    data_dict = data_dicts[int(name)]
    concrete = abstract.create_instance(data_dict)

    concrete._mpisppy_node_list = [
        mpisppy.scenario_tree.ScenarioNode(
            name="ROOT",
            cond_prob=1.0,
            stage=1,
            cost_expression=concrete.first_stage_cost,
            nonant_list=[concrete.is_active_depot],
            scen_model=concrete,
        )
    ]

    return concrete
