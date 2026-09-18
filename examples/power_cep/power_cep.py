###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2026, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyomo.environ as pyo

from mpisppy.utils import sputils

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

# Local example modules must be imported after MODULE_DIR is added to sys.path.
import scenario_generator as scen_gen  # noqa: E402

DEFAULT_DATA_FILE = Path(__file__).with_name("system_data.json")


def _load_system_data(data_path: str | None = None) -> dict:
    path = Path(data_path) if data_path is not None else DEFAULT_DATA_FILE
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scenario_names_creator(num_scens, start=None):
    if start is None:
        start = 0
    return [f"scen{i}" for i in range(start, start + num_scens)]


def inparser_adder(cfg):
    cfg.num_scens_required()
    cfg.add_to_config(
        "power_cep_data_path",
        description="path to the deterministic system data JSON",
        domain=str,
        default=str(DEFAULT_DATA_FILE),
    )
    cfg.add_to_config(
        "power_cep_seed",
        description="base seed for stochastic scenario generation",
        domain=int,
        default=0,
    )


def kw_creator(cfg):
    return {
        "system_data": _load_system_data(cfg.get("power_cep_data_path", str(DEFAULT_DATA_FILE))),
        "num_scens": cfg.get("num_scens", None),
        "seed": cfg.get("power_cep_seed", 0),
    }


def _build_model(scenario_name, system_data, num_scens, seed):
    scenario_data = scen_gen.generate_scenario_data(
        scenario_name=scenario_name,
        system_data=system_data,
        num_scens=num_scens,
        seed=seed,
    )

    model = pyo.ConcreteModel(name=scenario_name)

    rep_hours = list(system_data["representative_hours"])
    existing_generators = list(system_data["existing_generators"].keys())
    candidate_techs = list(system_data["candidate_technologies"].keys())

    model.H = pyo.Set(initialize=rep_hours, ordered=True)
    model.G_EXIST = pyo.Set(initialize=existing_generators, ordered=True)
    model.NEW_TECH = pyo.Set(initialize=candidate_techs, ordered=True)

    model.hour_weight = pyo.Param(
        model.H,
        initialize=lambda m, h: float(system_data["hour_weights"][h]),
        within=pyo.NonNegativeReals,
    )

    model.existing_capacity = pyo.Param(
        model.G_EXIST,
        initialize=lambda m, g: float(system_data["existing_generators"][g]["capacity_mw"]),
        within=pyo.NonNegativeReals,
    )
    model.existing_var_cost = pyo.Param(
        model.G_EXIST,
        initialize=lambda m, g: float(system_data["existing_generators"][g]["variable_cost_per_mwh"]),
        within=pyo.NonNegativeReals,
    )
    model.existing_tech = pyo.Param(
        model.G_EXIST,
        initialize=lambda m, g: system_data["existing_generators"][g]["technology"],
        within=pyo.Any,
    )

    model.candidate_var_cost = pyo.Param(
        model.NEW_TECH,
        initialize=lambda m, t: float(system_data["candidate_technologies"][t]["variable_cost_per_mwh"]),
        within=pyo.NonNegativeReals,
    )
    model.load_shedding_cost = pyo.Param(
        initialize=float(system_data["system_parameters"]["load_shedding_cost_per_mwh"]),
        within=pyo.NonNegativeReals,
    )

    gas_block_mw = float(system_data["candidate_technologies"]["gas"]["block_mw"])
    gas_max_blocks = int(system_data["candidate_technologies"]["gas"].get("max_build_blocks", 0))
    wind_max_build = float(system_data["candidate_technologies"]["wind"].get("max_build_mw", 0.0))
    solar_max_build = float(system_data["candidate_technologies"]["solar"].get("max_build_mw", 0.0))

    model.BuildGasUnits = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0, gas_max_blocks))
    model.BuildWindMW = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, wind_max_build))
    model.BuildSolarMW = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, solar_max_build))

    model.DispatchExisting = pyo.Var(model.G_EXIST, model.H, domain=pyo.NonNegativeReals)
    model.DispatchNewGas = pyo.Var(model.H, domain=pyo.NonNegativeReals)
    model.DispatchNewWind = pyo.Var(model.H, domain=pyo.NonNegativeReals)
    model.DispatchNewSolar = pyo.Var(model.H, domain=pyo.NonNegativeReals)
    model.LoadShedding = pyo.Var(model.H, domain=pyo.NonNegativeReals)

    def demand_rule(_m, h):
        return float(scenario_data["demand"][h])

    model.demand = pyo.Param(model.H, initialize=demand_rule, within=pyo.NonNegativeReals)

    def existing_avail_rule(_m, g, h):
        return float(scenario_data["availability_existing"][g][h])

    model.existing_availability = pyo.Param(
        model.G_EXIST,
        model.H,
        initialize=existing_avail_rule,
        within=pyo.NonNegativeReals,
    )

    def new_avail_rule(_m, t, h):
        return float(scenario_data["availability_new"][t][h])

    model.new_availability = pyo.Param(
        model.NEW_TECH,
        model.H,
        initialize=new_avail_rule,
        within=pyo.NonNegativeReals,
    )

    def existing_cap_rule(m, g, h):
        return m.DispatchExisting[g, h] <= m.existing_capacity[g] * m.existing_availability[g, h]

    model.ExistingCapacityLimit = pyo.Constraint(model.G_EXIST, model.H, rule=existing_cap_rule)

    def gas_cap_rule(m, h):
        return m.DispatchNewGas[h] <= gas_block_mw * m.BuildGasUnits * m.new_availability["gas", h]

    model.NewGasCapacityLimit = pyo.Constraint(model.H, rule=gas_cap_rule)

    def wind_cap_rule(m, h):
        return m.DispatchNewWind[h] <= m.BuildWindMW * m.new_availability["wind", h]

    model.NewWindCapacityLimit = pyo.Constraint(model.H, rule=wind_cap_rule)

    def solar_cap_rule(m, h):
        return m.DispatchNewSolar[h] <= m.BuildSolarMW * m.new_availability["solar", h]

    model.NewSolarCapacityLimit = pyo.Constraint(model.H, rule=solar_cap_rule)

    def balance_rule(m, h):
        existing_supply = sum(m.DispatchExisting[g, h] for g in m.G_EXIST)
        new_supply = m.DispatchNewGas[h] + m.DispatchNewWind[h] + m.DispatchNewSolar[h]
        return existing_supply + new_supply + m.LoadShedding[h] == m.demand[h]

    model.PowerBalance = pyo.Constraint(model.H, rule=balance_rule)

    gas_capex_annualized = float(system_data["candidate_technologies"]["gas"]["annualized_cost_per_block"])
    wind_capex_annualized = float(system_data["candidate_technologies"]["wind"]["annualized_cost_per_mw"])
    solar_capex_annualized = float(system_data["candidate_technologies"]["solar"]["annualized_cost_per_mw"])

    model.FirstStageCost = pyo.Expression(
        expr=(gas_capex_annualized * model.BuildGasUnits
              + wind_capex_annualized * model.BuildWindMW
              + solar_capex_annualized * model.BuildSolarMW)
    )

    model.SecondStageCost = pyo.Expression(
        expr=sum(
            model.hour_weight[h]
            * (
                sum(model.existing_var_cost[g] * model.DispatchExisting[g, h] for g in model.G_EXIST)
                + model.candidate_var_cost["gas"] * model.DispatchNewGas[h]
                + model.candidate_var_cost["wind"] * model.DispatchNewWind[h]
                + model.candidate_var_cost["solar"] * model.DispatchNewSolar[h]
                + model.load_shedding_cost * model.LoadShedding[h]
            )
            for h in model.H
        )
    )

    model.TotalCost = pyo.Objective(expr=model.FirstStageCost + model.SecondStageCost, sense=pyo.minimize)

    sputils.attach_root_node(model, model.FirstStageCost, [model.BuildGasUnits, model.BuildWindMW, model.BuildSolarMW])
    model._mpisppy_probability = scenario_data["probability"]
    return model


def scenario_creator(scenario_name, system_data=None, num_scens=None, seed=0):
    if system_data is None:
        system_data = _load_system_data()
    if num_scens is None:
        num_scens = 3
    return _build_model(scenario_name, system_data, num_scens, seed)


def scenario_denouement(rank, scenario_name, scenario):
    if scenario_name != "scen0":
        return
    print("Power CEP investment solution summary")
    print(f"  gas blocks built: {pyo.value(scenario.BuildGasUnits):.0f}")
    print(f"  wind MW built:    {pyo.value(scenario.BuildWindMW):.2f}")
    print(f"  solar MW built:   {pyo.value(scenario.BuildSolarMW):.2f}")
    print(f"  first-stage cost:  {pyo.value(scenario.FirstStageCost):.2f}")
