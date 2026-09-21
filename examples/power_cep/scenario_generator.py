###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2026, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScenarioTemplate:
    demand_scale: float
    renewable_scale: float
    gas_availability_shift: float
    demand_hour_multipliers: dict[str, float]
    wind_hour_multipliers: dict[str, float]
    solar_hour_multipliers: dict[str, float]


_TEMPLATES = (
    ScenarioTemplate(
        demand_scale=1.08,
        renewable_scale=0.90,
        gas_availability_shift=-0.01,
        demand_hour_multipliers={
            "winter_night": 1.00,
            "winter_day": 1.02,
            "spring_day": 1.03,
            "summer_afternoon_peak": 1.14,
            "summer_evening_peak": 1.12,
            "fall_shoulder_day": 1.01,
        },
        wind_hour_multipliers={
            "winter_night": 0.95,
            "winter_day": 0.92,
            "spring_day": 0.90,
            "summer_afternoon_peak": 0.84,
            "summer_evening_peak": 0.98,
            "fall_shoulder_day": 0.91,
        },
        solar_hour_multipliers={
            "winter_night": 0.00,
            "winter_day": 0.88,
            "spring_day": 0.95,
            "summer_afternoon_peak": 0.98,
            "summer_evening_peak": 0.62,
            "fall_shoulder_day": 0.90,
        },
    ),
    ScenarioTemplate(
        demand_scale=1.00,
        renewable_scale=1.00,
        gas_availability_shift=0.00,
        demand_hour_multipliers={
            "winter_night": 0.98,
            "winter_day": 1.00,
            "spring_day": 1.01,
            "summer_afternoon_peak": 1.06,
            "summer_evening_peak": 1.05,
            "fall_shoulder_day": 1.00,
        },
        wind_hour_multipliers={
            "winter_night": 1.00,
            "winter_day": 0.98,
            "spring_day": 1.00,
            "summer_afternoon_peak": 0.92,
            "summer_evening_peak": 1.00,
            "fall_shoulder_day": 0.96,
        },
        solar_hour_multipliers={
            "winter_night": 0.00,
            "winter_day": 1.00,
            "spring_day": 1.00,
            "summer_afternoon_peak": 1.00,
            "summer_evening_peak": 0.70,
            "fall_shoulder_day": 0.98,
        },
    ),
    ScenarioTemplate(
        demand_scale=0.93,
        renewable_scale=1.12,
        gas_availability_shift=0.01,
        demand_hour_multipliers={
            "winter_night": 0.94,
            "winter_day": 0.95,
            "spring_day": 0.96,
            "summer_afternoon_peak": 1.00,
            "summer_evening_peak": 0.98,
            "fall_shoulder_day": 0.95,
        },
        wind_hour_multipliers={
            "winter_night": 1.06,
            "winter_day": 1.04,
            "spring_day": 1.08,
            "summer_afternoon_peak": 1.02,
            "summer_evening_peak": 1.10,
            "fall_shoulder_day": 1.05,
        },
        solar_hour_multipliers={
            "winter_night": 0.00,
            "winter_day": 1.08,
            "spring_day": 1.12,
            "summer_afternoon_peak": 1.10,
            "summer_evening_peak": 0.82,
            "fall_shoulder_day": 1.04,
        },
    ),
)


def _template_for_index(index: int) -> ScenarioTemplate:
    return _TEMPLATES[index % len(_TEMPLATES)]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _scenario_rng(seed: int, scenario_index: int) -> np.random.Generator:
    return np.random.default_rng(seed + 1009 * scenario_index)


def generate_scenario_data(
    scenario_name: str,
    system_data: dict,
    num_scens: int,
    seed: int = 0,
) -> dict:
    """Create one scenario worth of stochastic data for the power CEP example."""
    from mpisppy.utils import sputils

    scenario_index = sputils.extract_num(scenario_name)
    template = _template_for_index(scenario_index)
    rng = _scenario_rng(seed, scenario_index)

    rep_hours = list(system_data["representative_hours"])
    existing = deepcopy(system_data["existing_generators"])
    candidates = deepcopy(system_data["candidate_technologies"])

    demand = {}
    availability_existing = {g: {} for g in existing}
    availability_new = {tech: {} for tech in candidates}

    for hour in rep_hours:
        demand_noise = 1.0 + rng.uniform(-0.04, 0.04)
        demand[hour] = 1000.0 * template.demand_scale * template.demand_hour_multipliers[hour] * demand_noise

        for gen_name, gen_data in existing.items():
            tech = gen_data["technology"]
            base_avail = float(gen_data["availability_profile"][hour])
            if tech == "gas":
                shifted = base_avail + template.gas_availability_shift + rng.uniform(-0.01, 0.01)
                availability_existing[gen_name][hour] = _clamp(shifted, 0.75, 1.0)
            else:
                perturb = 1.0 + rng.uniform(-0.03, 0.03)
                availability_existing[gen_name][hour] = _clamp(
                    base_avail * template.renewable_scale * perturb *
                    (template.wind_hour_multipliers[hour] if tech == "wind" else template.solar_hour_multipliers[hour]),
                    0.0,
                    1.0,
                )

        # New-build availability follows the same broad regime as the existing fleet.
        wind_base = 0.0
        solar_base = 0.0
        for gen_data in existing.values():
            if gen_data["technology"] == "wind":
                wind_base = float(gen_data["availability_profile"][hour])
            elif gen_data["technology"] == "solar":
                solar_base = float(gen_data["availability_profile"][hour])

        availability_new["gas"][hour] = _clamp(
            0.96 + template.gas_availability_shift + rng.uniform(-0.01, 0.01),
            0.75,
            1.0,
        )
        availability_new["wind"][hour] = _clamp(
            wind_base * template.renewable_scale * template.wind_hour_multipliers[hour] * (1.0 + rng.uniform(-0.03, 0.03)),
            0.0,
            1.0,
        )
        availability_new["solar"][hour] = _clamp(
            solar_base * template.renewable_scale * template.solar_hour_multipliers[hour] * (1.0 + rng.uniform(-0.03, 0.03)),
            0.0,
            1.0,
        )

    return {
        "scenario_name": scenario_name,
        "probability": 1.0 / float(num_scens),
        "demand": demand,
        "availability_existing": availability_existing,
        "availability_new": availability_new,
    }
