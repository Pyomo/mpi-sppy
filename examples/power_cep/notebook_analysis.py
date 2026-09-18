###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2026, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pyomo.environ as pyo

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

# Local example modules must be imported after MODULE_DIR is added to sys.path.
import notebook_solution_io as solio  # noqa: E402
import power_cep as pc  # noqa: E402
import scenario_generator as scen_gen  # noqa: E402
import tune_power_cep as tune  # noqa: E402


def solve_ev(system_data: dict, num_scens: int, seed: int = 0, solver_name: str = "highs"):
    ev_data = tune.avg_scenario_data(system_data, num_scens, seed)
    model = tune.build_model_from_data("EV", system_data, ev_data)
    tune.solve_model(model, solver_name)
    builds = tune.collect_builds(model)
    return {
        "model": model,
        "builds": builds,
        "builds_mw": _builds_to_mw(builds, system_data),
        "objective": pyo.value(model.TotalCost),
        "scenario_data": ev_data,
    }


def _builds_to_mw(builds: dict[str, float], system_data: dict) -> dict[str, float]:
    gas_block_mw = float(system_data["candidate_technologies"]["gas"]["block_mw"])
    return {
        "gas": float(builds.get("gas", 0.0)) * gas_block_mw,
        "wind": float(builds.get("wind", 0.0)),
        "solar": float(builds.get("solar", 0.0)),
    }


def solve_ev_and_rp(system_data: dict, num_scens: int, seed: int = 0, solver_name: str = "highs"):
    ev = solve_ev(system_data, num_scens, seed, solver_name)
    rp = solve_rp(system_data, num_scens, seed, solver_name)
    return {"ev": ev, "rp": rp}


def solve_rp(system_data: dict, num_scens: int, seed: int = 0, solver_name: str = "highs"):
    scenario_names = pc.scenario_names_creator(num_scens)
    from mpisppy.opt.ef import ExtensiveForm
    ef_obj = ExtensiveForm(
        {"solver": solver_name},
        scenario_names,
        pc.scenario_creator,
        scenario_creator_kwargs={"system_data": system_data, "num_scens": num_scens, "seed": seed},
    )
    ef_obj.solve_extensive_form()
    root_model = ef_obj.local_scenarios[scenario_names[0]]
    return {
        "ef": ef_obj,
        "builds": tune.collect_builds(root_model),
        "builds_mw": _builds_to_mw(tune.collect_builds(root_model), system_data),
        "objective": pyo.value(ef_obj.ef.EF_Obj),
    }


def extract_generation_rows_from_model(system_data: dict, scenario_name: str, model) -> list[dict]:
    tech_by_unit = {
        unit_name: unit_data["technology"]
        for unit_name, unit_data in system_data["existing_generators"].items()
    }
    hours = list(system_data["representative_hours"])
    rows = []
    for hour in hours:
        accum = defaultdict(float)
        for unit in model.G_EXIST:
            accum[tech_by_unit[str(unit)]] += float(pyo.value(model.DispatchExisting[unit, hour]))
        accum["gas"] += float(pyo.value(model.DispatchNewGas[hour]))
        accum["wind"] += float(pyo.value(model.DispatchNewWind[hour]))
        accum["solar"] += float(pyo.value(model.DispatchNewSolar[hour]))
        accum["load_shedding"] += float(pyo.value(model.LoadShedding[hour]))
        rows.append(
            {
                "scenario": scenario_name,
                "hour": hour,
                "gas": accum["gas"],
                "wind": accum["wind"],
                "solar": accum["solar"],
                "load_shedding": accum["load_shedding"],
            }
        )
    return rows


def load_rp_solution(solution_base_name: str | Path, system_data: dict | None = None):
    builds = solio.load_first_stage_builds(solution_base_name)
    tree_solution = solio.load_tree_solution_directory(f"{solution_base_name}_soldir")
    if system_data is None:
        gas_block_mw = 50.0
    else:
        gas_block_mw = float(system_data["candidate_technologies"]["gas"]["block_mw"])
    builds_mw = {"gas": builds.get("gas", 0.0) * gas_block_mw, "wind": builds.get("wind", 0.0), "solar": builds.get("solar", 0.0)}
    return {"builds": builds, "builds_mw": builds_mw, "tree_solution": tree_solution}


def build_rp_generation_table(system_data: dict, solution_base_name: str | Path):
    rp = load_rp_solution(solution_base_name)
    return solio.aggregate_generation_by_type(system_data, rp["tree_solution"])


def compute_scenario_data_table(system_data: dict, num_scens: int, seed: int = 0):
    scenario_names = pc.scenario_names_creator(num_scens)
    return solio.build_demand_table(
        system_data,
        scenario_names,
        lambda sname: scen_gen.generate_scenario_data(sname, system_data, num_scens, seed),
    )


def solve_ws_and_eev(system_data: dict, ev_builds: dict[str, float], num_scens: int, seed: int = 0, solver_name: str = "highs"):
    scenario_names = pc.scenario_names_creator(num_scens)
    ws_rows = []
    eev_rows = []
    ws_generation_rows = []
    eev_generation_rows = []
    ws_cost = 0.0
    eev_cost = 0.0

    for sname in scenario_names:
        sdata = scen_gen.generate_scenario_data(sname, system_data, num_scens, seed)
        ws_model = pc.scenario_creator(sname, system_data=system_data, num_scens=num_scens, seed=seed)
        tune.solve_model(ws_model, solver_name)
        ws_total = pyo.value(ws_model.TotalCost)
        ws_cost += sdata["probability"] * ws_total

        ev_model = pc.scenario_creator(sname, system_data=system_data, num_scens=num_scens, seed=seed)
        ev_model.BuildGasUnits.fix(ev_builds["gas"])
        ev_model.BuildWindMW.fix(ev_builds["wind"])
        ev_model.BuildSolarMW.fix(ev_builds["solar"])
        tune.solve_model(ev_model, solver_name)
        ev_total = pyo.value(ev_model.TotalCost)
        eev_cost += sdata["probability"] * ev_total

        ws_rows.append(
            {
                "scenario": sname,
                "probability": sdata["probability"],
                "total_cost": ws_total,
                "load_shedding": sum(pyo.value(ws_model.LoadShedding[h]) for h in ws_model.H),
            }
        )
        eev_rows.append(
            {
                "scenario": sname,
                "probability": sdata["probability"],
                "total_cost": ev_total,
                "load_shedding": sum(pyo.value(ev_model.LoadShedding[h]) for h in ev_model.H),
            }
        )
        ws_generation_rows.extend(extract_generation_rows_from_model(system_data, sname, ws_model))
        eev_generation_rows.extend(extract_generation_rows_from_model(system_data, sname, ev_model))

    return {
        "ws_cost": ws_cost,
        "eev_cost": eev_cost,
        "ws_rows": ws_rows,
        "eev_rows": eev_rows,
        "ws_generation_rows": ws_generation_rows,
        "eev_generation_rows": eev_generation_rows,
    }


def generation_rows_to_dataframe(rows):
    import pandas as pd

    return pd.DataFrame(rows)
