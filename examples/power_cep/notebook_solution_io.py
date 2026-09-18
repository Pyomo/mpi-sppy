###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2026, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

_INDEXED_NAME_RE = re.compile(r"^(?P<base>[A-Za-z_][A-Za-z0-9_]*)\[(?P<args>.*)\]$")


def read_name_value_csv(file_path: str | Path) -> dict[str, float]:
    path = Path(file_path)
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            # Variable names can themselves contain commas, e.g.
            # DispatchExisting[gas_existing_1,fall_shoulder_day].  The CSV
            # reader will split those names into multiple columns, so we treat
            # everything except the final field as the name.
            name = ",".join(part.strip() for part in row[:-1])
            value = float(row[-1])
            out[name] = value
    return out


def load_first_stage_solution(solution_base_name: str | Path) -> dict[str, float]:
    return read_name_value_csv(f"{solution_base_name}.csv")


def load_tree_solution_directory(solution_directory: str | Path) -> dict[str, dict[str, float]]:
    directory = Path(solution_directory)
    result: dict[str, dict[str, float]] = {}
    for csv_file in sorted(directory.glob("*.csv")):
        result[csv_file.stem] = read_name_value_csv(csv_file)
    return result


def parse_indexed_name(name: str):
    match = _INDEXED_NAME_RE.match(name)
    if match is None:
        return name, ()
    base = match.group("base")
    args = tuple(part.strip() for part in match.group("args").split(","))
    return base, args


def load_first_stage_builds(solution_base_name: str | Path) -> dict[str, float]:
    values = load_first_stage_solution(solution_base_name)
    return {
        "gas": values.get("BuildGasUnits", 0.0),
        "wind": values.get("BuildWindMW", 0.0),
        "solar": values.get("BuildSolarMW", 0.0),
    }


def build_existing_capacity_pie(system_data: dict) -> list[dict]:
    rows = []
    for gen_name, gen_data in system_data["existing_generators"].items():
        rows.append(
            {
                "unit": gen_name,
                "technology": gen_data["technology"],
                "capacity_mw": float(gen_data["capacity_mw"]),
            }
        )
    return rows


def build_demand_table(system_data: dict, scenario_names: list[str], scenario_data_creator) -> list[dict]:
    rows = []
    for sname in scenario_names:
        sdata = scenario_data_creator(sname)
        for hour in system_data["representative_hours"]:
            rows.append(
                {
                    "scenario": sname,
                    "hour": hour,
                    "demand_mw": float(sdata["demand"][hour]),
                    "probability": float(sdata["probability"]),
                }
            )
    return rows


def aggregate_generation_by_type(system_data: dict, tree_solution: dict[str, dict[str, float]]) -> list[dict]:
    tech_by_unit = {
        unit_name: unit_data["technology"]
        for unit_name, unit_data in system_data["existing_generators"].items()
    }
    hours = list(system_data["representative_hours"])
    rows = []

    for scenario_name, values in tree_solution.items():
        by_hour = {
            hour: defaultdict(float)
            for hour in hours
        }

        for var_name, value in values.items():
            base, args = parse_indexed_name(var_name)

            if base == "DispatchExisting" and len(args) == 2:
                unit, hour = args
                tech = tech_by_unit.get(unit)
                if tech is not None:
                    by_hour[hour][tech] += value
            elif base == "DispatchNewGas" and len(args) == 1:
                by_hour[args[0]]["gas"] += value
            elif base == "DispatchNewWind" and len(args) == 1:
                by_hour[args[0]]["wind"] += value
            elif base == "DispatchNewSolar" and len(args) == 1:
                by_hour[args[0]]["solar"] += value
            elif base == "LoadShedding" and len(args) == 1:
                by_hour[args[0]]["load_shedding"] += value

        for hour in hours:
            row = {
                "scenario": scenario_name,
                "hour": hour,
                "gas": by_hour[hour]["gas"],
                "wind": by_hour[hour]["wind"],
                "solar": by_hour[hour]["solar"],
                "load_shedding": by_hour[hour]["load_shedding"],
            }
            rows.append(row)

    return rows
