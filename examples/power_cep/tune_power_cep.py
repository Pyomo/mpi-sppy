###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2026, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

"""Battery-style tuning report for the power capacity expansion example.

This script computes the same pedagogical metrics used in the battery sizing
teaching example:

- EV: expected-value problem
- RP: recourse problem / stochastic program
- WS: wait-and-see / perfect-information benchmark
- EEV: expected cost of applying the EV solution under uncertainty
- VSS: EEV - RP
- EVPI: RP - WS

The script is intentionally lightweight and parameterized so it can be used as a
small tuning harness while the example data are refined.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pyomo.environ as pyo

from mpisppy.opt.ef import ExtensiveForm

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
for p in (MODULE_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Local example modules must be imported after their directories are added to sys.path.
import power_cep as pc  # noqa: E402
import scenario_generator as scen_gen  # noqa: E402


def build_model_from_data(scenario_name: str, system_data: dict, scenario_data: dict) -> pyo.ConcreteModel:
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

    model.demand = pyo.Param(
        model.H,
        initialize=lambda m, h: float(scenario_data["demand"][h]),
        within=pyo.NonNegativeReals,
    )
    model.existing_availability = pyo.Param(
        model.G_EXIST,
        model.H,
        initialize=lambda m, g, h: float(scenario_data["availability_existing"][g][h]),
        within=pyo.NonNegativeReals,
    )
    model.new_availability = pyo.Param(
        model.NEW_TECH,
        model.H,
        initialize=lambda m, t, h: float(scenario_data["availability_new"][t][h]),
        within=pyo.NonNegativeReals,
    )

    model.ExistingCapacityLimit = pyo.Constraint(
        model.G_EXIST,
        model.H,
        rule=lambda m, g, h: m.DispatchExisting[g, h] <= m.existing_capacity[g] * m.existing_availability[g, h],
    )
    model.NewGasCapacityLimit = pyo.Constraint(
        model.H,
        rule=lambda m, h: m.DispatchNewGas[h] <= gas_block_mw * m.BuildGasUnits * m.new_availability["gas", h],
    )
    model.NewWindCapacityLimit = pyo.Constraint(
        model.H,
        rule=lambda m, h: m.DispatchNewWind[h] <= m.BuildWindMW * m.new_availability["wind", h],
    )
    model.NewSolarCapacityLimit = pyo.Constraint(
        model.H,
        rule=lambda m, h: m.DispatchNewSolar[h] <= m.BuildSolarMW * m.new_availability["solar", h],
    )
    model.PowerBalance = pyo.Constraint(
        model.H,
        rule=lambda m, h: sum(m.DispatchExisting[g, h] for g in m.G_EXIST)
        + m.DispatchNewGas[h] + m.DispatchNewWind[h] + m.DispatchNewSolar[h] + m.LoadShedding[h]
        == m.demand[h],
    )

    model.FirstStageCost = pyo.Expression(
        expr=float(system_data["candidate_technologies"]["gas"]["annualized_cost_per_block"]) * model.BuildGasUnits
        + float(system_data["candidate_technologies"]["wind"]["annualized_cost_per_mw"]) * model.BuildWindMW
        + float(system_data["candidate_technologies"]["solar"]["annualized_cost_per_mw"]) * model.BuildSolarMW
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
    return model


def solve_model(model: pyo.ConcreteModel, solver_name: str = "highs"):
    solver = pyo.SolverFactory(solver_name)
    results = solver.solve(model, tee=False)
    return results


def avg_scenario_data(system_data: dict, num_scens: int, seed: int) -> dict:
    sdata = [scen_gen.generate_scenario_data(sname, system_data, num_scens, seed) for sname in pc.scenario_names_creator(num_scens)]
    rep_hours = list(system_data["representative_hours"])
    demand = {h: sum(sd["demand"][h] for sd in sdata) / num_scens for h in rep_hours}
    availability_existing = {
        g: {h: sum(sd["availability_existing"][g][h] for sd in sdata) / num_scens for h in rep_hours}
        for g in system_data["existing_generators"]
    }
    availability_new = {
        t: {h: sum(sd["availability_new"][t][h] for sd in sdata) / num_scens for h in rep_hours}
        for t in system_data["candidate_technologies"]
    }
    return {
        "scenario_name": "average",
        "probability": 1.0,
        "demand": demand,
        "availability_existing": availability_existing,
        "availability_new": availability_new,
    }


def eval_builds(builds: dict[str, float], system_data: dict, scenario_data: dict) -> tuple[float, float]:
    model = build_model_from_data(scenario_data["scenario_name"], system_data, scenario_data)
    model.BuildGasUnits.fix(builds["gas"])
    model.BuildWindMW.fix(builds["wind"])
    model.BuildSolarMW.fix(builds["solar"])
    solve_model(model)
    return pyo.value(model.TotalCost), pyo.value(model.SecondStageCost)


def collect_builds(model: pyo.ConcreteModel) -> dict[str, float]:
    return {
        "gas": float(pyo.value(model.BuildGasUnits)),
        "wind": float(pyo.value(model.BuildWindMW)),
        "solar": float(pyo.value(model.BuildSolarMW)),
    }


def _collect_builds(model: pyo.ConcreteModel) -> dict[str, float]:
    # Backward-compatible alias for any notebook cells or old references.
    return collect_builds(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-data", default=str(pc.DEFAULT_DATA_FILE))
    parser.add_argument("--num-scens", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solver-name", default="highs")
    parser.add_argument("--load-shedding-multiplier", type=float, default=1.0)
    parser.add_argument("--gas-capex-multiplier", type=float, default=1.0)
    parser.add_argument("--wind-capex-multiplier", type=float, default=1.0)
    parser.add_argument("--solar-capex-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    with Path(args.system_data).open("r", encoding="utf-8") as f:
        system_data = json.load(f)

    system_data = copy.deepcopy(system_data)
    system_data["system_parameters"]["load_shedding_cost_per_mwh"] *= args.load_shedding_multiplier
    system_data["candidate_technologies"]["gas"]["annualized_cost_per_block"] *= args.gas_capex_multiplier
    system_data["candidate_technologies"]["wind"]["annualized_cost_per_mw"] *= args.wind_capex_multiplier
    system_data["candidate_technologies"]["solar"]["annualized_cost_per_mw"] *= args.solar_capex_multiplier

    scenario_names = pc.scenario_names_creator(args.num_scens)

    print("=== Power CEP tuning report ===")
    print(f"scenarios: {scenario_names}")
    print(f"solver: {args.solver_name}")
    print()

    # EV
    ev_data = avg_scenario_data(system_data, args.num_scens, args.seed)
    ev_model = build_model_from_data("EV", system_data, ev_data)
    solve_model(ev_model, args.solver_name)
    ev_builds = collect_builds(ev_model)
    ev_nominal = pyo.value(ev_model.TotalCost)
    print("EV solution")
    print(f"  builds: {ev_builds}")
    print(f"  nominal objective: {ev_nominal:,.2f}")
    print()

    # RP via EF
    ef = ExtensiveForm(
        {"solver": args.solver_name},
        scenario_names,
        pc.scenario_creator,
        scenario_creator_kwargs={"system_data": system_data, "num_scens": args.num_scens, "seed": args.seed},
    )
    ef.solve_extensive_form()
    rp_model = ef.local_scenarios[scenario_names[0]]
    rp_builds = _collect_builds(rp_model)
    rp_cost = pyo.value(ef.ef.EF_Obj)
    print("RP solution")
    print(f"  builds: {rp_builds}")
    print(f"  expected objective: {rp_cost:,.2f}")
    print()

    # WS and EEV
    ws_cost = 0.0
    eev_cost = 0.0
    per_scenario_rows = []
    for sname in scenario_names:
        sdata = scen_gen.generate_scenario_data(sname, system_data, args.num_scens, args.seed)
        smodel = pc.scenario_creator(sname, system_data=system_data, num_scens=args.num_scens, seed=args.seed)
        solve_model(smodel, args.solver_name)
        scen_cost = pyo.value(smodel.TotalCost)
        ws_cost += sdata["probability"] * scen_cost

        ev_smodel = pc.scenario_creator(sname, system_data=system_data, num_scens=args.num_scens, seed=args.seed)
        ev_smodel.BuildGasUnits.fix(ev_builds["gas"])
        ev_smodel.BuildWindMW.fix(ev_builds["wind"])
        ev_smodel.BuildSolarMW.fix(ev_builds["solar"])
        solve_model(ev_smodel, args.solver_name)
        ev_scen_cost = pyo.value(ev_smodel.TotalCost)
        eev_cost += sdata["probability"] * ev_scen_cost

        per_scenario_rows.append(
            {
                "scenario": sname,
                "prob": sdata["probability"],
                "WS_cost": scen_cost,
                "EV_eval_cost": ev_scen_cost,
                "load_shed_WS": sum(pyo.value(smodel.LoadShedding[h]) for h in smodel.H),
                "load_shed_EV": sum(pyo.value(ev_smodel.LoadShedding[h]) for h in ev_smodel.H),
            }
        )

    vss = eev_cost - rp_cost
    evpi = rp_cost - ws_cost

    print("Scenario evaluation")
    for row in per_scenario_rows:
        print(
            f"  {row['scenario']}: prob={row['prob']:.3f}, "
            f"WS={row['WS_cost']:,.2f}, EV-eval={row['EV_eval_cost']:,.2f}, "
            f"shed_WS={row['load_shed_WS']:.2f}, shed_EV={row['load_shed_EV']:.2f}"
        )
    print()

    print("=== Summary ===")
    print(f"EV nominal objective: {ev_nominal:,.2f}")
    print(f"EEV expected cost:     {eev_cost:,.2f}")
    print(f"RP objective:          {rp_cost:,.2f}")
    print(f"WS expected cost:      {ws_cost:,.2f}")
    print(f"VSS = EEV - RP:        {vss:,.2f}")
    print(f"EVPI = RP - WS:        {evpi:,.2f}")
    print()
    print("Design comparison")
    print(f"  EV builds: {ev_builds}")
    print(f"  RP builds: {rp_builds}")


if __name__ == "__main__":
    main()
